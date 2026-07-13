import Capacitor

/// Capacitor's local/custom native plugins (ones living directly in this app
/// target rather than an npm package) aren't auto-discovered — the generated
/// ios/App/App/capacitor.config.json's packageClassList only covers plugins
/// Capacitor's CLI found in node_modules. CameraLinkPlugin has to be
/// registered explicitly here, which is why Main.storyboard's view
/// controller class was repointed at this subclass instead of using
/// Capacitor's stock CAPBridgeViewController directly.
class MainViewController: CAPBridgeViewController {
    override func capacitorDidLoad() {
        bridge?.registerPluginInstance(CameraLinkPlugin())
        bridge?.registerPluginInstance(RawDecoderPlugin())
        bridge?.registerPluginInstance(PhotoSaverPlugin())
    }
}
