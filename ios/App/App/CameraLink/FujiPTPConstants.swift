import Foundation

/// Standard PTP operation codes (ISO 15740) plus the Fuji-specific ones we
/// need. Ported from filmkit's src/ptp/constants.ts — see that file's own
/// comments for how these were reverse-engineered (Wireshark captures
/// against the official Fujifilm X RAW STUDIO app, cross-referenced against
/// a real X100VI's 7 custom presets, 2026-03).
enum PTPOp {
    static let openSession: UInt16 = 0x1002
    static let closeSession: UInt16 = 0x1003
    static let getDeviceInfo: UInt16 = 0x1001
    static let getDevicePropValue: UInt16 = 0x1015
    static let setDevicePropValue: UInt16 = 0x1016
}

enum PTPResp {
    static let ok: UInt16 = 0x2001
    static let generalError: UInt16 = 0x2002
    static let sessionNotOpen: UInt16 = 0x2003
    static let invalidTransactionID: UInt16 = 0x2004
    static let operationNotSupported: UInt16 = 0x2005
    static let parameterNotSupported: UInt16 = 0x2006
    static let incompleteTransfer: UInt16 = 0x2007
    static let invalidStorageID: UInt16 = 0x2008
    static let invalidObjectHandle: UInt16 = 0x2009
    static let devicePropNotSupported: UInt16 = 0x200A
    static let sessionAlreadyOpen: UInt16 = 0x201E

    private static let names: [UInt16: String] = [
        ok: "OK", generalError: "GeneralError", sessionNotOpen: "SessionNotOpen",
        invalidTransactionID: "InvalidTransactionID", operationNotSupported: "OperationNotSupported",
        parameterNotSupported: "ParameterNotSupported", incompleteTransfer: "IncompleteTransfer",
        invalidStorageID: "InvalidStorageID", invalidObjectHandle: "InvalidObjectHandle",
        devicePropNotSupported: "DevicePropNotSupported", sessionAlreadyOpen: "SessionAlreadyOpen",
    ]

    /// Human-readable "0xNNNN (Name)" for logging/error messages. `0x0000` is
    /// what parseResponseCode returns when it couldn't parse anything at all
    /// (nil/empty/malformed responseData) — flagged distinctly since that's a
    /// framing bug on our end, not a real camera response.
    static func describe(_ code: UInt16) -> String {
        if code == 0 { return "0x0000 (no code parsed — check responseData framing)" }
        let hex = "0x" + String(format: "%04X", code)
        if let name = names[code] { return "\(hex) (\(name))" }
        return hex
    }
}

/// Custom preset (C1-C7) property IDs. D18C selects the active slot; D18D is
/// the preset name; D18E-D1A5 are the 24 actual recipe fields.
enum FujiPresetProp {
    static let slotSelector: UInt16 = 0xD18C
    static let presetName: UInt16 = 0xD18D
    static let imageSize: UInt16 = 0xD18E
    static let imageQuality: UInt16 = 0xD18F
    static let dynamicRangePercent: UInt16 = 0xD190
    static let unknownD191: UInt16 = 0xD191
    static let filmSimulation: UInt16 = 0xD192
    static let monoWC: UInt16 = 0xD193
    static let monoMG: UInt16 = 0xD194
    static let grainEffect: UInt16 = 0xD195
    static let colorChrome: UInt16 = 0xD196
    static let colorChromeFxBlue: UInt16 = 0xD197
    static let smoothSkin: UInt16 = 0xD198
    static let whiteBalance: UInt16 = 0xD199
    static let wbShiftR: UInt16 = 0xD19A
    static let wbShiftB: UInt16 = 0xD19B
    static let wbColorTempK: UInt16 = 0xD19C
    static let highlightTone: UInt16 = 0xD19D
    static let shadowTone: UInt16 = 0xD19E
    static let color: UInt16 = 0xD19F
    static let sharpness: UInt16 = 0xD1A0
    static let highIsoNR: UInt16 = 0xD1A1
    static let clarity: UInt16 = 0xD1A2
    static let longExpNR: UInt16 = 0xD1A3
    static let colorSpace: UInt16 = 0xD1A4
    static let unknownD1A5: UInt16 = 0xD1A5

    /// All 24 setting properties for a slot, D18E through D1A5 inclusive.
    static let settingsRange: [UInt16] = Array(0xD18E...0xD1A5)
}

/// Film simulation values as reported by the camera's preset properties.
/// Ported from filmkit's src/profile/enums.ts FilmSim.
enum FujiFilmSim {
    static let provia: Int = 0x01
    static let velvia: Int = 0x02
    static let astia: Int = 0x03
    static let proNegHi: Int = 0x04
    static let proNegStd: Int = 0x05
    static let monochrome: Int = 0x06
    static let monochromeYe: Int = 0x07
    static let monochromeR: Int = 0x08
    static let monochromeG: Int = 0x09
    static let sepia: Int = 0x0A
    static let classicChrome: Int = 0x0B
    static let acros: Int = 0x0C
    static let acrosYe: Int = 0x0D
    static let acrosR: Int = 0x0E
    static let acrosG: Int = 0x0F
    static let eterna: Int = 0x10
    static let classicNeg: Int = 0x11
    static let eternaBleach: Int = 0x12
    static let nostalgicNeg: Int = 0x13
    static let realaAce: Int = 0x14

    static let monochromeVariants: Set<Int> = [monochrome, monochromeYe, monochromeR, monochromeG]
    static let acrosVariants: Set<Int> = [acros, acrosYe, acrosR, acrosG]
}

/// White balance mode values as reported by the camera's preset properties.
/// Ported from filmkit's src/profile/enums.ts WBMode.
enum FujiWBMode {
    static let asShot: Int = 0x0000
    static let auto: Int = 0x0002
    static let daylight: Int = 0x0004
    static let incandescent: Int = 0x0006
    static let underwater: Int = 0x0008
    static let fluorescent1: Int = 0x8001
    static let fluorescent2: Int = 0x8002
    static let fluorescent3: Int = 0x8003
    static let shade: Int = 0x8006
    static let colorTemp: Int = 0x8007
    static let ambiencePriority: Int = 0x8021
}

/// Dynamic range is stored as a raw percentage (100/200/400) in preset
/// properties, unlike the d185 RAW-conversion profile's 1/2/3 enum.
enum FujiDynamicRange {
    static let dr100 = 100
    static let dr200 = 200
    static let dr400 = 400
}

/// Grain is a flat 1-5 enum in preset properties (strength × size combined),
/// unlike the d185 profile's separate strength/size byte packing.
enum FujiPresetGrain {
    static let off = 1
    static let weakSmall = 2
    static let strongSmall = 3
    static let weakLarge = 4
    static let strongLarge = 5
}

/// Off/Weak/Strong effects (Color Chrome, Color Chrome FX Blue, Smooth Skin)
/// are 1-indexed in preset properties: 1=Off, 2=Weak, 3=Strong (NOT 0/1/2).
enum FujiPresetEffect {
    static let off = 1
    static let weak = 2
    static let strong = 3
}

/// Smart-decode raw PTP property bytes: PTP string, int16, or int32.
/// Mirrors filmkit's decodePropValue in src/ptp/session.ts.
enum PTPPropValue {
    case number(Int)
    case string(String)

    static func decode(_ data: Data) -> PTPPropValue {
        if data.count >= 3 {
            let numChars = Int(data[data.startIndex])
            let expectedLen = 1 + numChars * 2
            if numChars >= 2, expectedLen == data.count || expectedLen == data.count + 1 {
                return .string(decodePTPString(data))
            }
        }
        if data.count == 2 {
            let v = data.readLE(Int16.self, at: 0)
            return .number(Int(v))
        }
        if data.count == 4 {
            let v = data.readLE(Int32.self, at: 0)
            return .number(Int(v))
        }
        if data.count == 1 {
            return .number(Int(data[data.startIndex]))
        }
        return .number(0)
    }

    var intValue: Int? {
        if case let .number(v) = self { return v }
        return nil
    }

    var stringValue: String? {
        if case let .string(v) = self { return v }
        return nil
    }
}

/// PTP string: uint8 numChars (including the null terminator), then
/// numChars × UCS-2LE code units. Mirrors PTPReader.str() in binary.ts.
func decodePTPString(_ data: Data) -> String {
    guard data.count >= 1 else { return "" }
    let numChars = Int(data[data.startIndex])
    guard numChars > 0 else { return "" }
    var scalars: [UInt16] = []
    var offset = 1
    for _ in 0..<numChars {
        guard offset + 2 <= data.count else { break }
        let ch = data.readLE(UInt16.self, at: offset)
        if ch != 0 { scalars.append(ch) }
        offset += 2
    }
    return String(utf16CodeUnits: scalars, count: scalars.count)
}

/// Pack a PTP string: length byte (char count including null terminator) +
/// UTF-16LE chars + null terminator. Mirrors packPTPString in binary.ts.
func encodePTPString(_ value: String) -> Data {
    if value.isEmpty { return Data([0]) }
    let units = Array(value.utf16)
    var out = Data()
    // Length byte counts UTF-16 code units, not Swift Characters — they can
    // differ (e.g. combining/accented characters), and the byte count below
    // must match exactly what's emitted or the camera rejects the value.
    out.append(UInt8(min(units.count + 1, 255)))
    for scalar in units {
        out.appendLE(scalar)
    }
    out.appendLE(UInt16(0))
    return out
}

/// Decode a ×10-encoded tone value (highlight/shadow/color/sharpness/clarity).
/// 0x8000/-32768 is a sentinel meaning "not set" → treated as 0.
func decodeFujiTone(_ raw: Int) -> Double {
    if raw == 0x8000 || raw == -32768 { return 0 }
    return Double(raw) / 10.0
}

/// Decode the Fuji proprietary High ISO NR encoding (NOT ×10, NOT linear).
/// Ported from filmkit's NR_DECODE table.
func decodeFujiNoiseReduction(_ raw: Int) -> Int {
    let u16 = raw & 0xFFFF
    switch u16 {
    case 0x8000: return -4
    case 0x7000: return -3
    case 0x4000: return -2
    case 0x3000: return -1
    case 0x2000: return 0
    case 0x1000: return 1
    case 0x0000: return 2
    case 0x6000: return 3
    case 0x5000: return 4
    default: return 0
    }
}
