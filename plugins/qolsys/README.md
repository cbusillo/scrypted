# Qolsys Panel Camera local prototype

Uses IQ Remote client certificates to request fresh built-in camera pictures
without Alarm.com. Tested controller protocol: IQ Panel 2+ firmware 2.8.1.
Pictures are captured only on request, cached for five seconds, and their exact
panel file and request record are removed after download.

Video is an explicitly enabled experiment: one viewer and one 12-second native
recording per request. It uses newly recorded H264 units, never replayed clips.
The panel pauses its camera motion detector while recording, and only when that
setting was already on and the panel is on mains power. Downloads resend the
complete growing MP4, so playback can arrive in bursts and lag. The preview
stops only its own recording and removes its exact file and metadata on
completion or viewer disconnect. If an alarm takes the panel camera first, the
preview never stops the panel's own recording; an independent process keeps
retrying the stop until our recording has ended. Alarm, arming, or power
changes abort the preview.

Do not use this prototype for continuous prebuffering, Scrypted NVR or HomeKit
Secure Video recording. HomeKit live playback and owner acceptance are still
qualification work. No existing accessory pairing or alarm controls are changed
by this source. There are no arming or disarming methods in this adapter.

Set the panel address and select **Pair IQ Remote**, then press Pair under
IQ Remote Devices on the physical panel. Preserve that identity's PKI files in
the plugin volume's `qolsys/pki` directory. Under **Extensions**, turn off
**Rebroadcast Plugin** before enabling video: Scrypted attaches it automatically,
and its default prebuffer would repeatedly request this bounded preview.
Direct request-based snapshots and previews remain available without it.
To test video, enable **Experimental
live preview** and select **Calibrate video** while disarmed. Calibration makes
one eight-second clip, saves only decoder parameters (including the measured
frame rate), and removes the clip. Decoder parameters are re-saved after each
preview, so a quality or firmware change does not silently corrupt video. The
snapshot cache interval is configurable; each snapshot writes and deletes a
file on the panel, so viewers share the most recent image. Snapshot requests
while preview is busy may need to wait until it ends. The preview status
returns to **Ready** after the panel and viewer are cleaned up. Slow viewers
have bounded write and socket-close waits.

This plugin depends on the released `qolsys-controller` (see
`src/requirements.txt`), which must include the camera snapshot API and Python
3.12 support. The package stays private until that release is published; the
release `1.9.11` lacks the camera API and requires Python 3.14.

Run `npm install`, then `npm run build`; output is `out/plugin.zip`. Tests of the
camera-only transport and lifecycle live under `tests/` and use synthetic media.
