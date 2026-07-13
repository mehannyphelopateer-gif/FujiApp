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

            // Flat/neutral by design: boostAmount 0 means no global tone
            // curve (linear response) instead of the decoder's own default
            // rendering, and exposure/sharpness/contrast/noise-reduction are
            // all zeroed too — this is what makes the output free of any
            // baked-in "look" for a recipe to stack on top of.
            filter.boostAmount = 0
            filter.exposure = 0
            filter.sharpnessAmount = 0
            filter.contrastAmount = 0
            if filter.isLuminanceNoiseReductionSupported { filter.luminanceNoiseReductionAmount = 0 }
            if filter.isColorNoiseReductionSupported { filter.colorNoiseReductionAmount = 0 }

            guard let outputImage = filter.outputImage else {
                DispatchQueue.main.async {
                    call.reject("RAW decode produced no image.")
                }
                return
            }

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
