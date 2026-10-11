package org.openjarvisproject.ojp;

import android.app.NativeActivity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.service.voice.VoiceInteractionSession;

/**
 * Started by the assist gesture (long-press home / swipe from the corner)
 * once this app is the Digital assistant app. Rather than drawing a
 * second UI as an overlay, it brings the normal chat screen forward.
 */
public class AssistSession extends VoiceInteractionSession {
    public AssistSession(Context context) {
        super(context);
    }

    @Override
    public void onShow(Bundle args, int showFlags) {
        super.onShow(args, showFlags);
        Intent intent = new Intent(getContext(), NativeActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_REORDER_TO_FRONT);
        startAssistantActivity(intent);
        hide();
    }
}
