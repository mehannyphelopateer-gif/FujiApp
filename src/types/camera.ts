/**
 * Raw property payload for one camera custom-setting slot (C1-C7), as
 * returned by the native CameraLinkPlugin. Keys are property IDs formatted
 * as "D18E".."D1A5" (see ios/App/App/CameraLink/CameraLinkPlugin.swift).
 * Values are numbers or PTP strings — decodeSlot.ts turns this into a
 * Recipe-shaped object using the same encoding filmkit reverse-engineered.
 */
export interface CameraSlotRaw {
  slot: number;
  name: string;
  properties: Record<string, number | string>;
}

export interface CameraConnectionStatus {
  connected: boolean;
  deviceName: string;
}
