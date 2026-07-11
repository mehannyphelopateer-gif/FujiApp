import Foundation
import ImageCaptureCore

enum FujiCameraError: Error, LocalizedError {
    case noCameraFound
    case controlNotAuthorized
    case sessionFailed(String)
    case ptpError(String)
    case notConnected

    var errorDescription: String? {
        switch self {
        case .noCameraFound: return "No camera found. Connect the camera via USB-C and set it to USB RAW CONV./BACKUP RESTORE mode."
        case .controlNotAuthorized: return "Camera access was denied. Enable it in Settings."
        case .sessionFailed(let msg): return "Failed to open camera session: \(msg)"
        case .ptpError(let msg): return "Camera command failed: \(msg)"
        case .notConnected: return "Not connected to a camera."
        }
    }
}

struct RawProp {
    let id: UInt16
    let bytes: Data
    let value: PTPPropValue
}

struct PresetData {
    let slot: Int
    let name: String
    let settings: [RawProp]
}

/// Owns the ImageCaptureCore camera connection and speaks raw PTP over
/// `ICCameraDevice.requestSendPTPCommand`. This is the same PTP container
/// format and Fuji property map filmkit uses over WebUSB — ImageCaptureCore
/// is just a different transport underneath (see FujiPTPConstants.swift and
/// PTPContainer.swift for the ported protocol details).
///
/// NOTE — first-hardware-test caveat: exactly how `responseData`/
/// `ptpResponseData` map onto "response container" vs "raw data payload" in
/// ImageCaptureCore's completion isn't documented beyond Apple's terse header
/// comments. `parsePTPResponse` below handles both possible interpretations
/// defensively and logs raw byte lengths at each step — check the Xcode
/// console on first run against real hardware if reads come back empty or
/// garbled, since this is the one part of this file that's a best-informed
/// guess rather than a confirmed spec.
final class FujiCameraSession: NSObject {
    private let deviceBrowser = ICDeviceBrowser()
    private var camera: ICCameraDevice?
    private var transactionCounter: UInt32 = 1

    private var connectContinuation: CheckedContinuation<Void, Error>?
    private var sessionOpenContinuation: CheckedContinuation<Void, Error>?
    private var deviceReadyContinuation: CheckedContinuation<Void, Never>?

    var isConnected: Bool { camera != nil }
    var deviceName: String { camera?.name ?? "Unknown camera" }

    // MARK: - Connect / disconnect

    func connect() async throws {
        deviceBrowser.delegate = self
        deviceBrowser.browsedDeviceTypeMask = .camera

        let authStatus = deviceBrowser.controlAuthorizationStatus
        if authStatus == .denied || authStatus == .restricted {
            throw FujiCameraError.controlNotAuthorized
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.connectContinuation = continuation
            self.deviceBrowser.start()

            // Give the browser a window to find an already-connected camera;
            // if nothing shows up, fail rather than hang forever.
            DispatchQueue.main.asyncAfter(deadline: .now() + 8) { [weak self] in
                guard let self, let cont = self.connectContinuation else { return }
                self.connectContinuation = nil
                cont.resume(throwing: FujiCameraError.noCameraFound)
            }
        }

        guard let camera else { throw FujiCameraError.noCameraFound }
        camera.delegate = self

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            self.sessionOpenContinuation = continuation
            camera.requestOpenSession()
        }

        // deviceDidBecomeReadyWithCompleteContentCatalog must fire before PTP
        // commands are safe to send (per Apple's iOS 13.4 release-note caveat).
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            self.deviceReadyContinuation = continuation
        }
    }

    func disconnect() {
        if let camera {
            camera.requestCloseSession()
        }
        deviceBrowser.stop()
        camera = nil
    }

    // MARK: - Raw PTP

    private func nextTransactionId() -> UInt32 {
        defer { transactionCounter += 1 }
        return transactionCounter
    }

    /// GetDevicePropValue-style read: command phase only, camera replies with
    /// a data phase (the property bytes) then a response phase (the status code).
    func readProp(_ propId: UInt16) async throws -> RawProp? {
        let (code, data) = try await sendCommand(opcode: PTPOp.getDevicePropValue, params: [UInt32(propId)])
        guard code == PTPResp.ok, !data.isEmpty else { return nil }
        return RawProp(id: propId, bytes: data, value: PTPPropValue.decode(data))
    }

    private func sendCommand(opcode: UInt16, params: [UInt32]) async throws -> (code: UInt16, data: Data) {
        guard let camera else { throw FujiCameraError.notConnected }
        let transactionId = nextTransactionId()
        let commandContainer = PTPContainer.packCommand(code: opcode, transactionId: transactionId, params: params)

        return try await withCheckedThrowingContinuation { continuation in
            camera.requestSendPTPCommand(commandContainer, outData: nil) { responseData, ptpResponseData, error in
                if let error {
                    continuation.resume(throwing: FujiCameraError.ptpError(error.localizedDescription))
                    return
                }
                let code = FujiCameraSession.parseResponseCode(responseData)
                let data = FujiCameraSession.parseResponsePayload(ptpResponseData)
                continuation.resume(returning: (code, data))
            }
        }
    }

    /// SetDevicePropValue-style write: command phase, then a data phase
    /// carrying `bytes`, then a response phase with the status code.
    func writeProp(_ propId: UInt16, bytes: Data) async throws -> Bool {
        guard let camera else { throw FujiCameraError.notConnected }
        let transactionId = nextTransactionId()
        let commandContainer = PTPContainer.packCommand(code: PTPOp.setDevicePropValue, transactionId: transactionId, params: [UInt32(propId)])

        return try await withCheckedThrowingContinuation { continuation in
            camera.requestSendPTPCommand(commandContainer, outData: bytes) { responseData, _, error in
                if let error {
                    continuation.resume(throwing: FujiCameraError.ptpError(error.localizedDescription))
                    return
                }
                let code = FujiCameraSession.parseResponseCode(responseData)
                continuation.resume(returning: code == PTPResp.ok)
            }
        }
    }

    private static func parseResponseCode(_ responseData: Data?) -> UInt16 {
        guard let responseData else { return 0 }
        if let container = PTPContainer.unpack(responseData) {
            return container.code
        }
        // Fallback: some ImageCaptureCore versions may return just the raw
        // 2-byte code rather than a full container.
        if responseData.count == 2 {
            return responseData.readLE(UInt16.self, at: 0)
        }
        return 0
    }

    private static func parseResponsePayload(_ ptpResponseData: Data?) -> Data {
        guard let ptpResponseData, !ptpResponseData.isEmpty else { return Data() }
        // If this parses as a well-formed DATA container, unwrap it;
        // otherwise assume it's already the raw, unwrapped payload.
        if let container = PTPContainer.unpack(ptpResponseData), container.type == PTPContainerType.data.rawValue {
            return container.data
        }
        return ptpResponseData
    }

    // MARK: - Preset scan (read-only)

    /// Reads the name + all 24 settings for a single custom slot (1-7).
    /// Selecting the slot (writing D18C) is itself a property write — this
    /// mirrors filmkit's scanPresets() in src/ptp/session.ts.
    func readPreset(slot: Int) async throws -> PresetData {
        let selected = try await writeProp(FujiPresetProp.slotSelector, bytes: packU16(UInt16(slot)))
        guard selected else { throw FujiCameraError.ptpError("Failed to select slot \(slot)") }

        try? await Task.sleep(nanoseconds: 100_000_000) // 100ms, matches filmkit's post-slot-select delay

        let nameProp = try await readProp(FujiPresetProp.presetName)
        let name = nameProp?.value.stringValue ?? "(slot \(slot))"

        var settings: [RawProp] = []
        for propId in FujiPresetProp.settingsRange {
            if let prop = try await readProp(propId) {
                settings.append(prop)
            }
        }

        return PresetData(slot: slot, name: name, settings: settings)
    }

    /// Reads all 7 custom slots in order. Read-only — does not restore/change
    /// the camera's currently-selected slot afterward on purpose, matching
    /// Phase 2's scope (write path + slot-restore is Phase 3).
    func scanPresets() async throws -> [PresetData] {
        var results: [PresetData] = []
        for slot in 1...7 {
            let preset = try await readPreset(slot: slot)
            results.append(preset)
        }
        return results
    }
}

private func packU16(_ value: UInt16) -> Data {
    var v = value.littleEndian
    return Data(bytes: &v, count: 2)
}

// MARK: - ICDeviceBrowserDelegate

extension FujiCameraSession: ICDeviceBrowserDelegate {
    func deviceBrowser(_ browser: ICDeviceBrowser, didAdd device: ICDevice, moreComing: Bool) {
        guard camera == nil, let cameraDevice = device as? ICCameraDevice else { return }
        camera = cameraDevice
        connectContinuation?.resume(returning: ())
        connectContinuation = nil
    }

    func deviceBrowser(_ browser: ICDeviceBrowser, didRemove device: ICDevice, moreGoing: Bool) {
        if device === camera {
            camera = nil
        }
    }
}

// MARK: - ICCameraDeviceDelegate

extension FujiCameraSession: ICCameraDeviceDelegate {
    func device(_ device: ICDevice, didOpenSessionWithError error: Error?) {
        if let error {
            sessionOpenContinuation?.resume(throwing: FujiCameraError.sessionFailed(error.localizedDescription))
        } else {
            sessionOpenContinuation?.resume(returning: ())
        }
        sessionOpenContinuation = nil
    }

    func device(_ device: ICDevice, didCloseSessionWithError error: Error?) {
        // No-op — disconnect() doesn't wait on this.
    }

    func didRemove(_ device: ICDevice) {
        camera = nil
    }

    func deviceDidBecomeReady(_ device: ICDevice) {
        // ICDeviceDelegate's generic ready signal. For cameras we wait on the
        // more specific deviceDidBecomeReady(withCompleteContentCatalog:)
        // below, but resolve here too in case a given firmware never sends
        // that one.
        deviceReadyContinuation?.resume()
        deviceReadyContinuation = nil
    }

    func deviceDidBecomeReady(withCompleteContentCatalog device: ICCameraDevice) {
        deviceReadyContinuation?.resume()
        deviceReadyContinuation = nil
    }

    func cameraDevice(_ camera: ICCameraDevice, didAdd items: [ICCameraItem]) {}
    func cameraDevice(_ camera: ICCameraDevice, didRemove items: [ICCameraItem]) {}
    func cameraDevice(_ camera: ICCameraDevice, didRenameItems items: [ICCameraItem]) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceiveThumbnail thumbnail: CGImage?, for item: ICCameraItem, error: Error?) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceiveMetadata metadata: [AnyHashable: Any]?, for item: ICCameraItem, error: Error?) {}
    func cameraDeviceDidChangeCapability(_ camera: ICCameraDevice) {}
    func cameraDevice(_ camera: ICCameraDevice, didReceivePTPEvent eventData: Data) {}
    func cameraDeviceDidRemoveAccessRestriction(_ device: ICDevice) {}
    func cameraDeviceDidEnableAccessRestriction(_ device: ICDevice) {}
}
