# FujiApp

## Cross-device workflow

The user works on this project from multiple devices/VS Code instances. To keep them in sync:

- **At the start of any work session**, run `git fetch origin` and check `git log HEAD..origin/master --oneline`. If it's non-empty, pull before making any changes — another device may have pushed newer work.
- **After finishing a task** (build/typecheck passes), commit and push to `origin/master` so any other device can pick up the latest state. Don't leave finished work uncommitted locally.
- If local has uncommitted changes when you'd otherwise pull, stash (`git stash -u`) first, pull, then reapply — don't discard work.

## Deployment

This deploys as a **Cloudflare Worker with static assets** (not classic Cloudflare Pages — that product isn't available for new projects on this account).

- `wrangler.toml`: `main = "worker/index.ts"`, `[assets] directory = "./dist"` — deploy with `npx wrangler deploy`, **not** `wrangler pages deploy`.
- API routes live in `worker/index.ts` (routes `/api/recommend-recipe` and `/api/trip-recipe-chat` to `worker/handlers/*.ts`), falling back to `env.ASSETS.fetch(request)` for everything else. There is no `functions/` Pages-Functions directory — don't recreate one.
- `ANTHROPIC_API_KEY` is set as a Worker secret (`npx wrangler secret put ANTHROPIC_API_KEY`), not a dashboard build variable.
- Live at **https://fuji-recipes.com** (custom domain) and `https://fujifilmrecipes.mehannyphelopateer.workers.dev` (the account's `workers.dev` subdomain is fixed and can't be renamed — use the custom domain instead).
- The dashboard's Git-triggered "Workers Builds" auto-deploy pipeline may be unreliable (build-token issues); deploying directly via `npx wrangler deploy` from the CLI is the reliable path and is what's been used so far.

## iOS app (Capacitor)

The web app is also wrapped as a native iOS app via Capacitor, for personal sideload only (not App Store) — primarily to reach `ExternalAccessory`/native APIs a browser can't (e.g. the camera cable-link feature). `ios/` is Capacitor/Xcode-generated scaffolding; `ios/App/App/public` is a build copy of `dist/` and is gitignored/regenerated, same as `dist/` itself.

- `capacitor.config.ts`: `appId: "com.fujirecipes.app"`, `webDir: "dist"`.
- Rebuild + resync the native project after any web change: `npm run build && npx cap sync ios`.
- Open in Xcode: `npx cap open ios` (or open `ios/App/App.xcworkspace` directly).
- Simulator builds don't need code signing (`xcodebuild ... -sdk iphonesimulator`). Running on a physical iPhone needs the device owner's Apple ID selected as the signing team in Xcode's Signing & Capabilities tab — free personal-team signing expires/needs re-signing every 7 days unless upgraded to a paid Apple Developer account.
- No CocoaPods — Capacitor 8 uses Swift Package Manager (`ios/App/CapApp-SPM`) for its native dependencies.
