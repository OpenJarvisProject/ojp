package org.openjarvisproject.ojp;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.IBinder;
import android.os.PowerManager;

/**
 * Keeps Open Jarvis alive in the background, so the loaded model isn't thrown away when the app is
 * left, and replies keep being written with the screen off. It shows one silent, low-priority
 * notification. Nothing here talks to the network or any Google service.
 */
public class KeepAliveService extends Service {
    static final String ACTION_STOP = "org.openjarvisproject.ojp.STOP_KEEP_ALIVE";
    private static final int ID = 4711;
    private static PowerManager.WakeLock busyLock;

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent != null && ACTION_STOP.equals(intent.getAction())) {
            PhoneActions.setKeepAlive(this, false);
            stopForeground(true);
            stopSelf();
            return START_NOT_STICKY;
        }
        NotificationManager nm = (NotificationManager) getSystemService(NOTIFICATION_SERVICE);
        NotificationChannel ch = new NotificationChannel("keepalive", "Running in the background", NotificationManager.IMPORTANCE_MIN);
        ch.setDescription("Keeps Open Jarvis ready so it answers faster. Turn it off in Open Jarvis settings.");
        ch.setShowBadge(false);
        nm.createNotificationChannel(ch);
        Intent open = getPackageManager().getLaunchIntentForPackage(getPackageName());
        PendingIntent show = PendingIntent.getActivity(this, 0, open, PendingIntent.FLAG_IMMUTABLE);
        PendingIntent stop = PendingIntent.getService(this, 1, new Intent(this, KeepAliveService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE);
        Notification n = new Notification.Builder(this, "keepalive")
                .setSmallIcon(android.R.drawable.ic_menu_info_details)
                .setContentTitle("Open Jarvis is ready")
                .setContentText("Running quietly so it can answer quickly.")
                .setContentIntent(show)
                .setOngoing(true)
                .setShowWhen(false)
                .addAction(new Notification.Action.Builder(null, "Stop", stop).build())
                .build();
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        } else {
            startForeground(ID, n);
        }
        return START_STICKY;
    }

    /** Hold the CPU awake while a reply is being written (at most ten minutes at a time). */
    static synchronized void busy(Context c, boolean on) {
        if (on) {
            if (busyLock == null) {
                PowerManager pm = (PowerManager) c.getSystemService(Context.POWER_SERVICE);
                busyLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ojp:reply");
                busyLock.setReferenceCounted(false);
            }
            busyLock.acquire(10 * 60 * 1000L);
        } else if (busyLock != null && busyLock.isHeld()) {
            busyLock.release();
        }
    }
}
