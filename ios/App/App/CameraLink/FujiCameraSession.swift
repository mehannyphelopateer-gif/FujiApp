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

struct DeviceInfo {
    let model: String
    let raw: String
}

/// Owns the ImageCaptureCore camera connection and speaks raw PTP over
/// `ICCameraDevice.requestSendPTPCommand`. This is the same PTP container
/// format and Fuji property map filmkit uses over WebUSB — ImageCaptureCore
/// is just a different transport underneath (see FujiPTPConstants.swift and
/// PTPContainer.swift for the ported protocol details).
///
/// DIAGNOSTIC INSTRUMENTATION: this is the one part of the whole feature
/// that's a best-informed guess rather than a confirmed spec (exactly how
/// `responseData`/`ptpResponseData` map onto "response container" vs "raw
/// data payload" isn't documented beyond Apple's terse header comments), so
/// every PTP round-trip logs full hex dumps via NSLog AND embeds them in the
/// thrown error message — the latter so the raw bytes show up directly in
/// the debug UI without needing an Xcode console attached, since testing
/// happens on real hardware I don't have access to.
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

    /// Plain GetDeviceInfo (0x1001, no params, no outgoing data phase) — the
    /// simplest possible PTP round-trip. Use this to check whether raw PTP
    /// passthrough works AT ALL over ImageCaptureCore, isolated from any
    /// question about the Fuji-specific slot-select write.
    func getDeviceInfo() async throws -> DeviceInfo {
        let (code, _, data) = try await sendCommand(opcode: PTPOp.getDeviceInfo, params: [])
        guard code == PTPResp.ok else {
            throw FujiCameraError.ptpError("GetDeviceInfo returned \(PTPResp.describe(code)), data=\(data.count) bytes [\(data.hexPrefix())]")
        }
        // Don't fully parse the DeviceInfo dataset here — just prove the
        // round-trip works and show what came back.
        return DeviceInfo(model: camera?.name ?? "?", raw: "\(data.count) bytes: \(data.hexPrefix(64))")
    }

    /// GetDevicePropValue-style read: command phase only, camera replies with
    /// a data phase (the property bytes) then a response phase (the status code).
    func readProp(_ propId: UInt16) async throws -> RawProp? {
        let (code, _, data) = try await sendCommand(opcode: PTPOp.getDevicePropValue, params: [UInt32(propId)])
        guard code == PTPResp.ok, !data.isEmpty else { return nil }
        return RawProp(id: propId, bytes: data, value: PTPPropValue.decode(data))
    }

    /// `outData` carries a command's DATA phase when sending TO the camera
    /// (e.g. the ObjectInfo dataset, or a RAF's raw bytes) — `nil` for the
    /// simple property-read case this was originally written for.
    /// `timeoutSeconds` matters much more here than it used to: this is now
    /// also used for multi-megabyte transfers (RAF upload, converted-JPEG
    /// download) where a stuck completion callback would otherwise hang
    /// forever, indistinguishable from "still transferring."
    private func sendCommand(
        opcode: UInt16,
        params: [UInt32],
        outData: Data? = nil,
        timeoutSeconds: TimeInterval = 15
    ) async throws -> (code: UInt16, params: [UInt32], data: Data) {
        guard let camera else { throw FujiCameraError.notConnected }
        let transactionId = nextTransactionId()
        let commandContainer = PTPContainer.packCommand(code: opcode, transactionId: transactionId, params: params)
        NSLog(
            "[CameraLink] -> sendCommand opcode=0x%04X params=%@ tid=%u command=%@ outData=%d bytes",
            opcode, params, transactionId, commandContainer.hexPrefix(), outData?.count ?? 0
        )

        // Local per-call guard against double-resuming the continuation —
        // the timeout and the completion callback race, and exactly one of
        // them must win. Same accepted-risk pattern as connect()'s
        // continuation-cancel above (no lock; both closures are expected to
        // run on the main queue in practice).
        final class ResumeGuard { var didResume = false }
        let guardBox = ResumeGuard()

        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.main.asyncAfter(deadline: .now() + timeoutSeconds) {
                guard !guardBox.didResume else { return }
                guardBox.didResume = true
                continuation.resume(throwing: FujiCameraError.ptpError("opcode=0x\(String(opcode, radix: 16)) timed out after \(timeoutSeconds)s"))
            }

            camera.requestSendPTPCommand(commandContainer, outData: outData) { responseData, ptpResponseData, error in
                guard !guardBox.didResume else { return }
                guardBox.didResume = true

                NSLog(
                    "[CameraLink] <- sendCommand opcode=0x%04X error=%@ responseData(%d)=%@ ptpResponseData(%d)=%@",
                    opcode,
                    error?.localizedDescription ?? "nil",
                    responseData.count,
                    responseData.hexPrefix(),
                    ptpResponseData.count,
                    ptpResponseData.hexPrefix()
                )
                if let error {
                    continuation.resume(throwing: FujiCameraError.ptpError("opcode=0x\(String(opcode, radix: 16)) native error: \(error.localizedDescription)"))
                    return
                }
                // Confirmed empirically against real hardware (GetDeviceInfo
                // test, 2026-07): `ptpResponseData` is the raw PTP RESPONSE
                // container (status code + params) and `responseData` is the
                // DATA phase payload — the reverse of Apple's parameter-name
                // implication, but that's what the actual bytes show.
                let code = FujiCameraSession.parseResponseCode(ptpResponseData)
                let responseParams = FujiCameraSession.parseResponseParams(ptpResponseData)
                let data = FujiCameraSession.parseResponsePayload(responseData)
                continuation.resume(returning: (code, responseParams, data))
            }
        }
    }

    // MARK: - RAW conversion: upload + object transfer

    /// Uploads a .RAF's raw bytes to the camera via the two-step Fuji vendor
    /// sequence (SendObjectInfo then SendObject2). Do this once per file —
    /// re-converting with a different recipe only needs readProp/writeProp
    /// on 0xD185/0xD183 plus the object-handle dance below, not another
    /// upload. Timeouts are generous (large file, real USB transfer time).
    func uploadRaf(_ rafData: Data) async throws {
        let objectInfo = FujiObjectTransfer.buildObjectInfoDataset(fileSize: UInt32(rafData.count), filename: "FUP_FILE.dat")
        let (infoCode, _, _) = try await sendCommand(opcode: PTPOp.sendObjectInfo, params: [0, 0, 0], outData: objectInfo, timeoutSeconds: 30)
        guard infoCode == PTPResp.ok else {
            throw FujiCameraError.ptpError("SendObjectInfo failed: \(PTPResp.describe(infoCode))")
        }

        let (sendCode, _, _) = try await sendCommand(opcode: PTPOp.sendObject2, params: [], outData: rafData, timeoutSeconds: 90)
        guard sendCode == PTPResp.ok else {
            throw FujiCameraError.ptpError("SendObject2 failed: \(PTPResp.describe(sendCode))")
        }
    }

    /// Lists every object handle currently on the camera/card — used to spot
    /// the newly-created converted JPEG after triggering a RAW conversion
    /// (by diffing against a baseline taken before the trigger).
    func listObjectHandles() async throws -> [UInt32] {
        let (code, _, data) = try await sendCommand(opcode: PTPOp.getObjectHandles, params: [0xFFFFFFFF, 0, 0])
        guard code == PTPResp.ok else {
            throw FujiCameraError.ptpError("GetObjectHandles failed: \(PTPResp.describe(code))")
        }
        return FujiObjectTransfer.parseObjectHandleArray(data)
    }

    /// Downloads an object's full bytes (the converted JPEG) by handle.
    func downloadObject(handle: UInt32) async throws -> Data {
        let (code, _, data) = try await sendCommand(opcode: PTPOp.getObject, params: [handle], timeoutSeconds: 60)
        guard code == PTPResp.ok else {
            throw FujiCameraError.ptpError("GetObject failed: \(PTPResp.describe(code))")
        }
        return data
    }

    /// Deletes a temporary object (the converted JPEG) after downloading it,
    /// so repeated conversions don't leave junk files on the camera/card.
    /// Best-effort by design — callers should treat failure as non-fatal.
    @discardableResult
    func deleteObject(handle: UInt32) async throws -> Bool {
        let (code, _, _) = try await sendCommand(opcode: PTPOp.deleteObject, params: [handle])
        return code == PTPResp.ok
    }

    /// SetDevicePropValue-style write: command phase, then a data phase
    /// carrying `bytes`, then a response phase with the status code.
    func writeProp(_ propId: UInt16, bytes: Data) async throws -> Bool {
        guard let camera else { throw FujiCameraError.notConnected }
        let transactionId = nextTransactionId()
        let commandContainer = PTPContainer.packCommand(code: PTPOp.setDevicePropValue, transactionId: transactionId, params: [UInt32(propId)])
        NSLog("[CameraLink] -> writeProp id=0x%04X tid=%u command=%@ outData=%@", propId, transactionId, commandContainer.hexPrefix(), bytes.hexPrefix())

        return try await withCheckedThrowingContinuation { continuation in
            camera.requestSendPTPCommand(commandContainer, outData: bytes) { responseData, ptpResponseData, error in
                NSLog(
                    "[CameraLink] <- writeProp id=0x%04X error=%@ responseData(%d)=%@ ptpResponseData(%d)=%@",
                    propId,
                    error?.localizedDescription ?? "nil",
                    responseData.count,
                    responseData.hexPrefix(),
                    ptpResponseData.count,
                    ptpResponseData.hexPrefix()
                )
                if let error {
                    continuation.resume(throwing: FujiCameraError.ptpError("prop=0x\(String(propId, radix: 16)) native error: \(error.localizedDescription)"))
                    return
                }
                // See sendCommand()'s comment — ptpResponseData carries the
                // status code, responseData carries any payload.
                let code = FujiCameraSession.parseResponseCode(ptpResponseData)
                if code != PTPResp.ok {
                    let diag = "prop=0x\(String(propId, radix: 16)) \(PTPResp.describe(code)) responseData=\(responseData.hexPrefix()) ptpResponseData=\(ptpResponseData.hexPrefix())"
                    continuation.resume(throwing: FujiCameraError.ptpError(diag))
                    return
                }
                continuation.resume(returning: true)
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

    /// Extracts the response container's params (if any) from `ptpResponseData`
    /// — needed for object-transfer ops like SendObjectInfo, whose useful
    /// result (e.g. the assigned object handle) lives in the response params,
    /// not the data phase.
    private static func parseResponseParams(_ ptpResponseData: Data?) -> [UInt32] {
        guard let ptpResponseData, let container = PTPContainer.unpack(ptpResponseData) else { return [] }
        return container.params
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

    // MARK: - Preset write (Phase 3)

    /// Writes a full preset to a camera slot, then reads every successfully
    /// written property straight back and byte-compares it against what was
    /// sent. Mirrors filmkit's writePreset() in src/ptp/session.ts: slot
    /// selection and name-write failures are fatal (abort immediately),
    /// individual property write/verify failures are collected as
    /// non-fatal warnings — some properties (sentinels, read-only fields)
    /// are known to sometimes reject writes even on a healthy camera.
    ///
    /// `properties` must already be in the order the caller wants them sent
    /// — see src/lib/camera/encodeRecipe.ts, which replicates filmkit's
    /// documented write ordering (e.g. D19C must immediately follow D199).
    func writePreset(slot: Int, name: String, properties: [(id: UInt16, value: Int)]) async throws -> (ok: Bool, warnings: [String]) {
        let selected = try await writeProp(FujiPresetProp.slotSelector, bytes: packU16(UInt16(slot)))
        guard selected else {
            return (false, ["Failed to select slot \(slot)"])
        }
        try? await Task.sleep(nanoseconds: 100_000_000)

        do {
            let nameOk = try await writeProp(FujiPresetProp.presetName, bytes: encodePTPString(name))
            guard nameOk else { return (false, ["Failed to write preset name"]) }
        } catch {
            return (false, ["Failed to write preset name: \(error.localizedDescription)"])
        }

        var warnings: [String] = []
        var written: [UInt16: Data] = [:]
        for (id, value) in properties {
            let bytes = packPropertyValue(value)
            do {
                let ok = try await writeProp(id, bytes: bytes)
                if ok {
                    written[id] = bytes
                } else {
                    warnings.append("0x\(String(id, radix: 16)): write rejected [\(bytes.hexPrefix())]")
                }
            } catch {
                warnings.append("0x\(String(id, radix: 16)): \(error.localizedDescription)")
            }
        }

        // Verify name.
        if let verifyName = try await readProp(FujiPresetProp.presetName), let readName = verifyName.value.stringValue, readName != name {
            return (false, ["Name verify failed: wrote \"\(name)\" read \"\(readName)\""])
        }

        // Verify only the properties that reported a successful write.
        for (id, sentBytes) in written {
            guard let readBack = try await readProp(id) else { continue }
            if readBack.bytes != sentBytes {
                warnings.append("0x\(String(id, radix: 16)): verify mismatch — sent [\(sentBytes.hexPrefix())] read [\(readBack.bytes.hexPrefix())]")
            }
        }

        return (true, warnings)
    }
}

/// Packs a logical property value into its 2-byte little-endian wire form.
/// Non-negative inputs are taken as a raw UInt16 bit pattern directly (correct
/// both for small positive "signed" values like tone x10, and for raw enum
/// values that exceed Int16.max, e.g. WB ColorTemp mode = 0x8007 = 32775, or
/// the HighIsoNR sentinel 0x8000 = 32768). Negative inputs go through Int16's
/// two's-complement bit pattern (e.g. WB shift -9, shadow tone -20).
private func packPropertyValue(_ value: Int) -> Data {
    let u16: UInt16 = value >= 0 ? UInt16(value & 0xFFFF) : UInt16(bitPattern: Int16(clamping: value))
    var le = u16.littleEndian
    return Data(bytes: &le, count: 2)
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
