package org.openjarvisproject.ojp;

import android.content.Intent;
import android.os.RemoteException;
import android.speech.RecognitionService;
import android.speech.SpeechRecognizer;

/**
 * Exists only because a voice-interaction service must name a
 * RecognitionService. Pointing at our own stub instead of the Google
 * app's keeps the app free of any Google dependency, so it can still be
 * picked as the assistant on de-Googled phones. Speech recognition itself
 * will run in-app (Rust), not through this API.
 */
public class NoopRecognitionService extends RecognitionService {
    @Override
    protected void onStartListening(Intent intent, Callback callback) {
        try {
            callback.error(SpeechRecognizer.ERROR_CLIENT);
        } catch (RemoteException ignored) {
        }
    }

    @Override
    protected void onCancel(Callback callback) {
    }

    @Override
    protected void onStopListening(Callback callback) {
    }
}
