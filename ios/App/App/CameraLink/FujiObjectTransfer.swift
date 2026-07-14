import Foundation

/// Pure data-transformation helpers for the PTP object-transfer operations
/// (SendObjectInfo/SendObject2/GetObjectHandles) used to upload a .RAF to the
/// camera and retrieve back a converted JPEG. No camera/session/transport
/// state lives here — same spirit as PTPContainer.swift, just for the
/// object-info/handle-array datasets instead of the command/response
/// container itself.
enum FujiObjectTransfer {
    /// Builds the PTP ObjectInfo dataset that must accompany SendObjectInfo
    /// when uploading a .RAF. This is a standard PTP (ISO 15740) dataset
    /// layout, not Fuji-proprietary — only the ObjectFormat value (0xF802)
    /// and the opcode used to send it (Fuji's vendor 0x900C, see PTPOp) are
    /// camera-specific. Verified field-by-field against eggricesoy/filmkit's
    /// real, hardware-tested implementation of this exact upload.
    static func buildObjectInfoDataset(fileSize: UInt32, filename: String) -> Data {
        var data = Data()
        data.appendLE(UInt32(0))              // StorageID
        data.appendLE(fujiRafObjectFormat)     // ObjectFormat
        data.appendLE(UInt16(0))               // ProtectionStatus
        data.appendLE(fileSize)                 // CompressedSize
        data.appendLE(UInt16(0))               // ThumbFormat
        data.appendLE(UInt32(0))               // ThumbCompressedSize
        data.appendLE(UInt32(0))               // ThumbPixWidth
        data.appendLE(UInt32(0))               // ThumbPixHeight
        data.appendLE(UInt32(0))               // ImagePixWidth
        data.appendLE(UInt32(0))               // ImagePixHeight
        data.appendLE(UInt32(0))               // ImageBitDepth
        data.appendLE(UInt32(0))               // ParentObject
        data.appendLE(UInt16(0))               // AssociationType
        data.appendLE(UInt32(0))               // AssociationDesc
        data.appendLE(UInt32(0))               // SequenceNumber
        data.append(encodePTPString(filename)) // Filename
        data.append(UInt8(0))                  // CaptureDate (empty PTP string: 0 chars)
        data.append(UInt8(0))                  // ModificationDate (empty PTP string)
        data.append(UInt8(0))                  // Keywords (empty PTP string)
        return data
    }

    /// Parses a standard PTP object-handle array: a UInt32 count followed by
    /// that many UInt32 handles, all little-endian. Used to read the result
    /// of GetObjectHandles when polling for the camera's converted JPEG.
    static func parseObjectHandleArray(_ data: Data) -> [UInt32] {
        guard data.count >= 4 else { return [] }
        let count = Int(data.readLE(UInt32.self, at: 0))
        var handles: [UInt32] = []
        handles.reserveCapacity(count)
        var offset = 4
        for _ in 0..<count {
            guard offset + 4 <= data.count else { break }
            handles.append(data.readLE(UInt32.self, at: offset))
            offset += 4
        }
        return handles
    }

    /// Parses a standard PTP ObjectInfo dataset (the response payload of
    /// GetObjectInfo) — the mirror image of buildObjectInfoDataset above.
    /// Only the fields this app actually needs are extracted: the fixed-size
    /// fields up through SequenceNumber are skipped by known byte offsets,
    /// then Filename is read as a PTP string starting right after them.
    /// (CaptureDate/ModificationDate/Keywords, which follow Filename, aren't
    /// needed and aren't parsed.)
    static func parseObjectInfo(_ data: Data) -> (filename: String, size: UInt32, objectFormat: UInt16)? {
        let filenameOffset = 52
        guard data.count > filenameOffset else { return nil }
        let objectFormat = data.readLE(UInt16.self, at: 4)
        let size = data.readLE(UInt32.self, at: 8)
        let filename = decodePTPString(data.suffix(from: filenameOffset))
        return (filename, size, objectFormat)
    }
}
