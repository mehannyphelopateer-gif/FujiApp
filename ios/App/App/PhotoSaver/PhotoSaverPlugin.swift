import Foundation
import Capacitor
import Photos
import UIKit

/// Saves an image straight to the Photos library, bypassing the share
/// sheet entirely. navigator.share({ files: [...] }) — the web-standard
/// approach used on other platforms — doesn't reliably surface a
/// "Save Image" action on iOS when triggered from inside a Capacitor
/// WKWebView: the share sheet only offers that action for items the OS
/// recognizes as a real photo (via NSItemProvider/UIActivityItemProvider
/// conforming to the image UTI), and a bridged WKWebView's Web Share
/// implementation doesn't get the same native glue Safari itself gets, so
/// it falls back to generic file actions (Save to Files, AirDrop, etc.)
/// with no Photos option at all. Writing directly via PHPhotoLibrary
/// sidesteps that gap completely.
@objc(PhotoSaverPlugin)
public class PhotoSaverPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "PhotoSaverPlugin"
    public let jsName = "PhotoSaver"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "saveImage", returnType: CAPPluginReturnPromise)
    ]

    /// `data` is a base64-encoded JPEG (the WebGL canvas's exported frame).
    @objc func saveImage(_ call: CAPPluginCall) {
        guard let base64 = call.getString("data"),
              let data = Data(base64Encoded: base64),
              let image = UIImage(data: data) else {
            call.reject("Missing or invalid image data.")
            return
        }

        // .addOnly only needs "add to library" consent (NSPhotoLibraryAddUsageDescription),
        // not full read/write access to the whole Photos library.
        PHPhotoLibrary.requestAuthorization(for: .addOnly) { status in
            guard status == .authorized || status == .limited else {
                DispatchQueue.main.async {
                    call.reject("Photos access was denied. Enable it in Settings > FujiApp > Photos.")
                }
                return
            }

            PHPhotoLibrary.shared().performChanges({
                PHAssetChangeRequest.creationRequestForAsset(from: image)
            }, completionHandler: { success, error in
                DispatchQueue.main.async {
                    if success {
                        call.resolve(["saved": true])
                    } else {
                        call.reject(error?.localizedDescription ?? "Failed to save the image.")
                    }
                }
            })
        }
    }
}
