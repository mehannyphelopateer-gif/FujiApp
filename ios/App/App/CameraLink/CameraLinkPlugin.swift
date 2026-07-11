import Foundation
import Capacitor

/// Capacitor bridge for FujiCameraSession. Deliberately thin — all the
/// "what does property 0xD190 actually mean" logic lives in TypeScript
/// (src/lib/camera/decodeSlot.ts), using the exact same encoding tables
/// ported from filmkit. This plugin just moves raw property id → value pairs
/// across the JS bridge.
///
/// Phase 2 scope: read-only (connect, getStatus, scanPresets, disconnect).
/// No writeRecipeToSlot yet — that's Phase 3, once reads are confirmed
/// accurate against the real camera.
@objc(CameraLinkPlugin)
public class CameraLinkPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "CameraLinkPlugin"
    public let jsName = "CameraLink"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "connect", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "disconnect", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getStatus", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "scanPresets", returnType: CAPPluginReturnPromise),
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
