package org.openjarvisproject.ojp;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Rings when a reminder or timer set through {@link PhoneActions} comes due. */
public class ReminderReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context c, Intent intent) {
        String text = intent.getStringExtra("text");
        if (text == null) text = "Reminder";
        NotificationManager nm = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel("reminders", "Reminders and timers", NotificationManager.IMPORTANCE_HIGH));
        Intent open = c.getPackageManager().getLaunchIntentForPackage(c.getPackageName());
        PendingIntent pi = PendingIntent.getActivity(c, 0, open, PendingIntent.FLAG_IMMUTABLE);
        Notification n = new Notification.Builder(c, "reminders")
                .setSmallIcon(android.R.drawable.ic_popup_reminder)
                .setContentTitle("OpenJarvis")
                .setContentText(text)
                .setStyle(new Notification.BigTextStyle().bigText(text))
                .setContentIntent(pi)
                .setAutoCancel(true)
                .build();
        nm.notify((int) (System.currentTimeMillis() & 0x7fffffff), n);
    }
}
