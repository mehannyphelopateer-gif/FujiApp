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

    private static func presetToPayload(_ preset: PresetData) -> [String: Any] {
        var properties: [String: Any] = [:]
        for setting in preset.settings {
            let key = String(format: "D%X", setting.id)
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
