import Foundation
import Capacitor

/// Capacitor bridge for FujiCameraSession. Deliberately thin — all the
/// "what does property 0xD190 actually mean" logic lives in TypeScript
/// (src/lib/camera/decodeSlot.ts), using the exact same encoding tables
/// ported from filmkit. This plugin just moves raw property id → value pairs
/// across the JS bridge.
///
/// Phase 3: writeRecipeToSlot adds the write path on top of Phase 2's
/// read-only connect/getStatus/scanPresets/disconnect.
@objc(CameraLinkPlugin)
public class CameraLinkPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "CameraLinkPlugin"
    public let jsName = "CameraLink"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "connect", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "disconnect", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getStatus", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getDeviceInfo", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "scanPresets", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "writeRecipeToSlot", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "uploadRaf", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getRawProfile", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "setRawProfile", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "startRawConversion", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "listObjectHandles", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "downloadObject", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "deleteObject", returnType: CAPPluginReturnPromise),
    ]

    private let session = FujiCameraSession()

    @objc func connect(_ call: CAPPluginCall) {
        Task {
            do {
                try await session.connect()
                call.resolve(["connected": true, "deviceName": session.deviceName])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    @objc func disconnect(_ call: CAPPluginCall) {
        session.disconnect()
        call.resolve()
    }

    @objc func getStatus(_ call: CAPPluginCall) {
        call.resolve(["connected": session.isConnected, "deviceName": session.deviceName])
    }

    /// Diagnostic-only: plain GetDeviceInfo, no Fuji-specific properties
    /// involved. Isolates "does raw PTP passthrough work at all over
    /// ImageCaptureCore" from "does the Fuji slot-select write work" —
    /// see FujiCameraSession.getDeviceInfo()'s doc comment.
    @objc func getDeviceInfo(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        Task {
            do {
                let info = try await session.getDeviceInfo()
                call.resolve(["model": info.model, "raw": info.raw])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    @objc func scanPresets(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        Task {
            do {
                let presets = try await session.scanPresets()
                call.resolve(["slots": presets.map(Self.presetToPayload)])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    /// Writes a recipe to a camera slot. `properties` must be an ordered
    /// array of `{id: "D19D", value: -20}`-style objects (hex id string,
    /// signed/raw int value) — see src/lib/camera/encodeRecipe.ts, which
    /// owns both the property meaning and the required write order.
    @objc func writeRecipeToSlot(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        guard let slot = call.getInt("slot"), (1...7).contains(slot) else {
            call.reject("slot must be an integer 1-7.")
            return
        }
        let name = call.getString("name") ?? ""
        guard let rawProperties = call.getArray("properties", JSObject.self) else {
            call.reject("properties must be an array of {id, value} objects.")
            return
        }

        var properties: [(id: UInt16, value: Int)] = []
        for entry in rawProperties {
            guard let idHex = entry["id"] as? String, let id = UInt16(idHex, radix: 16) else { continue }
            guard let value = entry["value"] as? Int else { continue }
            properties.append((id: id, value: value))
        }

        Task {
            do {
                let result = try await session.writePreset(slot: slot, name: name, properties: properties)
                call.resolve(["ok": result.ok, "warnings": result.warnings])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    // MARK: - RAW conversion (Phase 1 scaffolding — see project plan)
    //
    // Deliberately thin, same philosophy as the rest of this file: these
    // methods just move base64 bytes and handle numbers across the bridge.
    // src/lib/camera/patchRawProfile.ts owns what the profile bytes mean.

    /// Uploads a .RAF's full bytes to the camera. Call once per file — do
    /// NOT call again just to try a different recipe on the same file.
    @objc func uploadRaf(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        guard let base64 = call.getString("data"), let data = Data(base64Encoded: base64) else {
            call.reject("data must be a base64-encoded .RAF file.")
            return
        }
        Task {
            do {
                try await session.uploadRaf(data)
                call.resolve(["ok": true])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    /// Reads the camera's current RAW-conversion profile (0xD185). Only
    /// returns valid data once a RAF has been uploaded via uploadRaf.
    @objc func getRawProfile(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        Task {
            do {
                guard let prop = try await session.readProp(FujiRawConvProp.profile) else {
                    call.reject("Camera returned no RAW conversion profile — is a .RAF loaded?")
                    return
                }
                call.resolve(["profile": prop.bytes.base64EncodedString(), "length": prop.bytes.count])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    /// Writes a patched RAW-conversion profile back (0xD185). `profile` must
    /// be the base64 of a full profile blob — normally the camera's own,
    /// as returned by getRawProfile, patched via patchRawProfile.ts.
    @objc func setRawProfile(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        guard let base64 = call.getString("profile"), let data = Data(base64Encoded: base64) else {
            call.reject("profile must be a base64-encoded profile blob.")
            return
        }
        Task {
            do {
                let ok = try await session.writeProp(FujiRawConvProp.profile, bytes: data)
                call.resolve(["ok": ok])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    /// Triggers RAW conversion using whatever profile is currently set
    /// (0xD183). Poll listObjectHandles afterward for the resulting JPEG.
    @objc func startRawConversion(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        Task {
            do {
                let ok = try await session.writeProp(FujiRawConvProp.startConversion, bytes: Data([0x00, 0x00]))
                call.resolve(["ok": ok])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    @objc func listObjectHandles(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        Task {
            do {
                let handles = try await session.listObjectHandles()
                call.resolve(["handles": handles.map { Int($0) }])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    @objc func downloadObject(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        guard let handle = call.getInt("handle") else {
            call.reject("handle must be an integer object handle.")
            return
        }
        Task {
            do {
                let data = try await session.downloadObject(handle: UInt32(handle))
                call.resolve(["data": data.base64EncodedString()])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    @objc func deleteObject(_ call: CAPPluginCall) {
        guard session.isConnected else {
            call.reject("Not connected. Call connect() first.")
            return
        }
        guard let handle = call.getInt("handle") else {
            call.reject("handle must be an integer object handle.")
            return
        }
        Task {
            do {
                let ok = try await session.deleteObject(handle: UInt32(handle))
                call.resolve(["ok": ok])
            } catch {
                call.reject(error.localizedDescription, nil, error)
            }
        }
    }

    private static func presetToPayload(_ preset: PresetData) -> [String: Any] {
        var properties: [String: Any] = [:]
        for setting in preset.settings {
            // Property IDs (0xD18C-0xD1A5) already start with "D" once
            // hex-formatted (e.g. 0xD196 -> "D196") — prepending a literal
            // "D" here produced "DD196", which never matched the "D196"-style
            // keys decodeSlot.ts looks for, so every decoded field silently
            // fell back to its default (confirmed against a real scan: raw
            // values like DD19A=-1/DD19B=-5 were genuinely being read
            // correctly, just filed under the wrong key).
            let key = String(format: "%X", setting.id)
            switch setting.value {
            case .number(let n): properties[key] = n
            case .string(let s): properties[key] = s
            }
        }
        return [
            "slot": preset.slot,
            "name": preset.name,
            "properties": properties,
        ]
    }
}
