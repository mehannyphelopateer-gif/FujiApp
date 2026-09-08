import Foundation
import Capacitor
import CoreImage
import UIKit

/// Decodes a Fujifilm .RAF file's actual sensor data via Apple's CIRAWFilter
/// (Core Image's built-in RAW demosaicer, with native Fuji X-Trans support —
/// confirmed against a real X-Pro1 sample RAF, 2026-07) rather than reading
/// the file's embedded JPEG preview. That preview is already rendered
/// through the camera's own JPEG engine — whatever film simulation/grain was
/// dialed in at capture is baked into its pixels and can't be undone. A real
/// RAW decode never goes through that engine at all, so applying a
/// *different* recipe afterward (src/lib/recipes/neutralize.ts's
/// computeRecipeAdjustment, with a `null` detected baseline) starts from a
/// genuinely clean base instead of stacking on top of the old recipe.
@objc(RawDecoderPlugin)
public class RawDecoderPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "RawDecoderPlugin"
    public let jsName = "RawDecoder"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "decodeNeutral", returnType: CAPPluginReturnPromise)
    ]

    private let ciContext = CIContext()

    /// `data` is the full .RAF file, base64-encoded (small enough files that
    /// the JS side can read it whole via FileReader — a 26MB X100VI RAF is a
    /// ~35MB base64 string, comfortably under the bridge's message limits).
    @objc func decodeNeutral(_ call: CAPPluginCall) {
        guard let base64 = call.getString("data"), let data = Data(base64Encoded: base64) else {
            call.reject("Missing or invalid RAW file data.")
            return
        }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            guard let filter = CIRAWFilter(imageData: data, identifierHint: nil) else {
                DispatchQueue.main.async {
                    call.reject("This device's RAW decoder couldn't open the file.")
                }
                return
            }

            // DIAGNOSTIC, added to investigate a real finding: Apple's newest
            // "RAW 9" CIRAWFilter decoder version ignores colorNoiseReductionAmount
            // entirely, replacing manual chroma-NR control with an automatic,
            // non-configurable CoreML denoising model (confirmed via Apple's
            // WWDC26 session 305 and its own colorNoiseReductionAmount docs).
            // If this device defaults to RAW 9, the colorNoiseReductionAmount
            // override below may be a partial or total no-op — this print
            // shows up in Xcode's console on a real device run and tells us
            // definitively which decoder version is actually active, and
            // which alternatives this file/device supports, before guessing
            // at an explicit decoderVersion override (an invalid version
            // string silently produces a nil output image per Apple's docs,
            // so this must be driven by the real reported values, not a
            // guess). See ~/.claude/plans/indexed-inventing-wren.md's Phase 3
            // "Round 11".
            print("[RawDecoder] supportedDecoderVersions=\(filter.supportedDecoderVersions) defaultDecoderVersion=\(filter.decoderVersion)")

            // Deliberately left at CIRAWFilter's own defaults (boostAmount 1,
            // i.e. its normal tone curve; default sharpness/contrast/NR).
            // An earlier version forced boostAmount to 0 for a "linear,
            // nothing baked in" image, on the theory that anything else
            // would count as a look stacked under the recipe. That backfired
            // in testing: linear response is a flat, low-contrast image very
            // different from a normal photo, and the shader pipeline below
            // (LUTs, tone curve, saturation, grain) was built and tuned
            // entirely against normally-contrasty camera-JPEG-style images —
            // it has no calibration for scene-linear input. Recipes applied
            // on top of the linear version didn't look like the recipe at
            // all. CIRAWFilter's default rendering is what's actually
            // "neutral" for this app's purposes: a normal-looking photo with
            // no Fuji-specific film simulation baked in (Fuji's film sim is
            // entirely a JPEG-engine construct that a RAW decode never
            // touches, regardless of these settings), matching the same kind
            // of input the pipeline already expects from any other JPEG.
            //
            // ONE explicit override, added after direct pixel analysis of a
            // real dramatically-underexposed photo (a museum statue under a
            // single warm spotlight) found the blue channel hard-clipped to
            // literally 0 in 46% of true-shadow pixels (luma < 30/255), and
            // under 3/255 in 90% of them — real color information a real
            // Fuji JPEG conversion of the same RAF clearly preserves (a real
            // X RAW Studio export of the identical file has meaningfully
            // higher, non-crushed blue values in the same shadow regions).
            // No white-balance math applied later in the WebGL shader can
            // ever recover a channel that's already exactly 0 by the time it
            // gets there — 0 times any gain is still 0. CIRAWFilter's
            // default color noise reduction is the most likely cause:
            // Apple's own documentation describes kCIInputColorNoiseReductionAmountKey
            // (0...1) as chroma-specific noise reduction, which in a very
            // dark, high-ISO-equivalent region can easily read a faint real
            // blue signal as noise and flatten it toward neutral gray/zero.
            // Disabling it trades a small amount of potential color-noise
            // speckle in shadows (arguably a fine trade for a film-
            // simulation app, which already has its own separate grain
            // effect) for not silently destroying real, recoverable color
            // data before any of Phase 1-3's calibrated pipeline ever sees
            // it. Confirmed on the real device (Round 11): this device only
            // supports decoder versions 7/8 (default 8) — NOT Apple's newest
            // "RAW 9", which replaces colorNoiseReductionAmount with a non-
            // configurable CoreML model. So colorNoiseReductionAmount is a
            // genuinely live control here, confirmed by a real, substantial
            // (though incomplete) improvement: true-shadow pixels with blue
            // hard-clipped to exactly 0 dropped from 46.1% to 23.2% after
            // this override alone. The remaining crushed pixels were spread
            // broadly across luma 0-29 (not just literal near-zero luma),
            // suggesting more real, recoverable headroom — so also
            // also disabling the separate luminanceNoiseReductionAmount
            // (independently documented, a real control at this decoder
            // version, 0...1 range) to eliminate every remaining
            // controllable noise-reduction pass between the sensor data and
            // this app's calibrated pipeline. (There is no separate general
            // `noiseReductionAmount` property on this SDK's typed
            // CIRAWFilter interface — confirmed by the compiler rejecting
            // it — only the luminance/color split above.) Trades some
            // potential visible grain/speckle in shadows for not smoothing
            // away real color data — an acceptable trade for a film-
            // simulation app that already has its own separate grain
            // effect. NOT YET VALIDATED against a real device retest — see
            // ~/.claude/plans/indexed-inventing-wren.md's Phase 3
            // "Round 12".
            filter.colorNoiseReductionAmount = 0.0
            filter.luminanceNoiseReductionAmount = 0.0

            // Round 13: disabling both noise-reduction controls plateaued at
            // ~23-24% of true-shadow pixels with blue hard-clipped to 0 —
            // confirmed NOT a sensor/physical floor by comparing against a
            // real X RAW Studio conversion of the identical RAF, which
            // shows only 5.5% in the same region (still real, recoverable
            // color data, just not surfacing through this decode).
            // boostShadowAmount is a separate, independently documented
            // property (default 1, range 0...2) specifically for
            // "lighten[ing] the shadow areas of the image... to lighten
            // details in shadows" — untouched until now. Raising it gives
            // shadow tones more room in the final 8-bit output before
            // whatever processing stage is currently rounding faint color
            // signal down to 0, on the theory that more headroom there
            // means less gets lost to quantization. NOT YET VALIDATED — see
            // ~/.claude/plans/indexed-inventing-wren.md's Phase 3
            // "Round 13".
            filter.boostShadowAmount = 2.0

            guard let rawOutput = filter.outputImage else {
                DispatchQueue.main.async {
                    call.reject("RAW decode produced no image.")
                }
                return
            }

            // Measured, not guessed: decoding the same sensor data twice —
            // once through this filter, once through the camera's own JPEG
            // engine (a RAF's embedded preview is that real Fuji rendering)
            // — and averaging saturation ((max-min)/max per pixel) over both
            // showed CIRAWFilter's default output sitting at ~0.51 against
            // Fuji's own ~0.44, a consistent ~12% gap. Their contrast/tone
            // range, by contrast, were nearly identical (Fuji stddev 57.2 vs
            // this filter's 53.4), which is why only saturation is corrected
            // here and not contrast/exposure. Reapplying that 0.876 factor
            // brought average saturation and RGB balance within about a
            // percent of the real Fuji rendering for that same test file —
            // still just one sample/one lighting condition, so treat this as
            // a starting point that may need revisiting against more of the
            // user's own real shots, not a final calibration.
            let colorControls = CIFilter(name: "CIColorControls")
            colorControls?.setValue(rawOutput, forKey: kCIInputImageKey)
            colorControls?.setValue(0.876, forKey: kCIInputSaturationKey)
            let outputImage = colorControls?.outputImage ?? rawOutput

            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
                  let cgImage = self.ciContext.createCGImage(
                    outputImage,
                    from: outputImage.extent,
                    format: .RGBA8,
                    colorSpace: colorSpace
                  ) else {
                DispatchQueue.main.async {
                    call.reject("Failed to render the decoded RAW image.")
                }
                return
            }

            let uiImage = UIImage(cgImage: cgImage)
            guard let jpegData = uiImage.jpegData(compressionQuality: 0.92) else {
                DispatchQueue.main.async {
                    call.reject("Failed to encode the decoded image.")
                }
                return
            }

            DispatchQueue.main.async {
                call.resolve(["data": jpegData.base64EncodedString()])
            }
        }
    }
}
