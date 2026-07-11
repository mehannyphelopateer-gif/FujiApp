import Foundation

/// PTP container types (ISO 15740).
enum PTPContainerType: UInt16 {
    case command = 0x0001
    case data = 0x0002
    case response = 0x0003
    case event = 0x0004
}

/// A parsed PTP container (command, data, or response phase).
///
/// Mirrors filmkit's `PTPContainerData`/`packContainer`/`unpackContainer`
/// (src/ptp/container.ts) — same 12-byte header layout, ported 1:1 since
/// ImageCaptureCore's `requestSendPTPCommand` expects/returns raw container
/// bytes, not a higher-level API.
struct PTPContainer {
    var type: UInt16
    var code: UInt16
    var transactionId: UInt32
    var params: [UInt32]
    var data: Data

    private static let headerSize = 12

    /// Pack a COMMAND-phase container: header + up to 5 uint32 params, no payload.
    static func packCommand(code: UInt16, transactionId: UInt32, params: [UInt32] = []) -> Data {
        pack(type: PTPContainerType.command.rawValue, code: code, transactionId: transactionId, params: params, data: Data())
    }

    /// Pack a DATA-phase container: header + raw payload, no params.
    static func packData(code: UInt16, transactionId: UInt32, data: Data) -> Data {
        pack(type: PTPContainerType.data.rawValue, code: code, transactionId: transactionId, params: [], data: data)
    }

    private static func pack(type: UInt16, code: UInt16, transactionId: UInt32, params: [UInt32], data: Data) -> Data {
        let clampedParams = Array(params.prefix(5))
        let paramsSize = clampedParams.count * 4
        let totalLength = UInt32(headerSize + paramsSize + data.count)

        var out = Data(capacity: Int(totalLength))
        out.appendLE(totalLength)
        out.appendLE(type)
        out.appendLE(code)
        out.appendLE(transactionId)
        for p in clampedParams { out.appendLE(p) }
        out.append(data)
        return out
    }

    /// Unpack a RESPONSE or DATA container returned from the camera.
    /// RESPONSE containers carry up to 5 uint32 params and no payload;
    /// DATA containers carry a payload and no params — same split as filmkit.
    static func unpack(_ raw: Data) -> PTPContainer? {
        guard raw.count >= headerSize else { return nil }

        let type = raw.readLE(UInt16.self, at: 4)
        let code = raw.readLE(UInt16.self, at: 6)
        let transactionId = raw.readLE(UInt32.self, at: 8)
        let rest = raw.subdata(in: headerSize..<raw.count)

        var params: [UInt32] = []
        var data = Data()

        if type == PTPContainerType.data.rawValue {
            data = rest
        } else if type == PTPContainerType.response.rawValue {
            var offset = 0
            while offset + 4 <= rest.count, params.count < 5 {
                params.append(rest.readLE(UInt32.self, at: offset))
                offset += 4
            }
        }

        return PTPContainer(type: type, code: code, transactionId: transactionId, params: params, data: data)
    }
}

extension Data {
    mutating func appendLE(_ value: UInt16) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 2))
    }

    mutating func appendLE(_ value: UInt32) {
        var v = value.littleEndian
        append(Data(bytes: &v, count: 4))
    }

    /// Little-endian read, byte-by-byte — avoids the unsafe-pointer generic
    /// closure form, which was tripping a Swift compiler internal error here.
    func readLE<T: FixedWidthInteger>(_ type: T.Type, at offset: Int) -> T {
        let byteCount = MemoryLayout<T>.size
        var result: T = 0
        for i in 0..<byteCount {
            let byte = self[self.startIndex + offset + i]
            result |= T(byte) << (8 * i)
        }
        return result
    }
}
