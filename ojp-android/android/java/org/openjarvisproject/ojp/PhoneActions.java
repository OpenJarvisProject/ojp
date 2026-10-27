package org.openjarvisproject.ojp;

import android.app.Activity;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.media.AudioManager;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.view.KeyEvent;

import java.util.List;
import java.util.Locale;

/**
 * What the assistant can do on the phone through plain Android APIs, with no extra permission and
 * no helper app. One entry point, {@link #invoke}, so the Rust side needs a single JNI call:
 * it returns "ok:<text>" or "err:<message>". Shizuku-backed actions ({@code shell} and friends)
 * are routed to {@link ShizukuBridge}, which is optional.
 */
public final class PhoneActions {
    private static Activity activity;

    private PhoneActions() {}

    /** Called once from the native side with the running activity. */
    public static void init(Activity a) {
        activity = a;
    }

    private static Context ctx() {
        return activity.getApplicationContext();
    }

    public static String invoke(String method, String arg) {
        try {
            return "ok:" + dispatch(method, arg == null ? "" : arg);
        } catch (Throwable t) {
            String m = t.getMessage();
            return "err:" + (m == null || m.isEmpty() ? t.toString() : m);
        }
    }

    private static String dispatch(String method, String arg) throws Exception {
        switch (method) {
            case "volume": {
                AudioManager am = (AudioManager) ctx().getSystemService(Context.AUDIO_SERVICE);
                int max = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC);
                int pct = Math.max(0, Math.min(100, Integer.parseInt(arg.trim())));
                am.setStreamVolume(AudioManager.STREAM_MUSIC, Math.round(max * pct / 100f), AudioManager.FLAG_SHOW_UI);
                return "";
            }
            case "mute": {
                AudioManager am = (AudioManager) ctx().getSystemService(Context.AUDIO_SERVICE);
                am.adjustStreamVolume(AudioManager.STREAM_MUSIC,
                        arg.equals("on") ? AudioManager.ADJUST_MUTE : AudioManager.ADJUST_UNMUTE, AudioManager.FLAG_SHOW_UI);
                return "";
            }
            case "media": {
                int code;
                switch (arg) {
                    case "play_pause": code = KeyEvent.KEYCODE_MEDIA_PLAY_PAUSE; break;
                    case "next": code = KeyEvent.KEYCODE_MEDIA_NEXT; break;
                    case "previous": code = KeyEvent.KEYCODE_MEDIA_PREVIOUS; break;
                    default: throw new Exception("unknown media action " + arg);
                }
                AudioManager am = (AudioManager) ctx().getSystemService(Context.AUDIO_SERVICE);
                am.dispatchMediaKeyEvent(new KeyEvent(KeyEvent.ACTION_DOWN, code));
                am.dispatchMediaKeyEvent(new KeyEvent(KeyEvent.ACTION_UP, code));
                return "";
            }
            case "open_app":
                return openApp(arg);
            case "open_url": {
                Uri uri = Uri.parse(arg.trim());
                String scheme = uri.getScheme();
                if (!"https".equals(scheme) && !"http".equals(scheme)) throw new Exception("only web links can be opened");
                activity.startActivity(new Intent(Intent.ACTION_VIEW, uri).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
                return "";
            }
            case "remind":
                return remind(arg);
            case "battery": {
                Intent b = ctx().registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
                if (b == null) throw new Exception("battery status isn't available");
                int level = b.getIntExtra(BatteryManager.EXTRA_LEVEL, -1);
                int scale = b.getIntExtra(BatteryManager.EXTRA_SCALE, 100);
                int status = b.getIntExtra(BatteryManager.EXTRA_STATUS, -1);
                String s = status == BatteryManager.BATTERY_STATUS_CHARGING ? "charging"
                        : status == BatteryManager.BATTERY_STATUS_FULL ? "full" : "not charging";
                return "battery " + Math.round(100f * level / scale) + "% (" + s + ")";
            }
            case "shizuku_state":
                return ShizukuBridge.state();
            case "shizuku_request":
                ShizukuBridge.requestPermission();
                return "";
            case "open_shizuku": {
                Intent i = ctx().getPackageManager().getLaunchIntentForPackage("moe.shizuku.privileged.api");
                if (i == null) throw new Exception("Shizuku isn't installed");
                activity.startActivity(i.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
                return "";
            }
            case "shell":
                return ShizukuBridge.run(arg);
            default:
                throw new Exception("unknown action " + method);
        }
    }

    /** Launch an installed app by the name shown under its icon (best match wins). */
    private static String openApp(String name) throws Exception {
        PackageManager pm = ctx().getPackageManager();
        Intent main = new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER);
        List<ResolveInfo> apps = pm.queryIntentActivities(main, 0);
        String q = name.trim().toLowerCase(Locale.ROOT);
        ResolveInfo best = null;
        String bestLabel = "";
        int bestScore = 0;
        for (ResolveInfo ri : apps) {
            String label = ri.loadLabel(pm).toString();
            String l = label.toLowerCase(Locale.ROOT);
            int score = l.equals(q) ? 4 : l.startsWith(q) ? 3 : l.contains(q) ? 2
                    : ri.activityInfo.packageName.toLowerCase(Locale.ROOT).contains(q.replace(" ", "")) ? 1 : 0;
            if (score > bestScore || (score == bestScore && score > 0 && label.length() < bestLabel.length())) {
                best = ri;
                bestLabel = label;
                bestScore = score;
            }
        }
        if (best == null) throw new Exception("I couldn't find an app called " + name);
        Intent launch = pm.getLaunchIntentForPackage(best.activityInfo.packageName);
        if (launch == null) throw new Exception(bestLabel + " can't be opened");
        activity.startActivity(launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        return bestLabel;
    }

    /** "dueMillis TAB text": ring an alarm-clock alarm (exact, no special permission) that posts a notification. */
    private static String remind(String arg) throws Exception {
        String[] p = arg.split("\t", 2);
        long due = Long.parseLong(p[0]);
        String text = p.length > 1 ? p[1] : "";
        Context c = ctx();
        AlarmManager am = (AlarmManager) c.getSystemService(Context.ALARM_SERVICE);
        Intent fire = new Intent(c, ReminderReceiver.class).putExtra("text", text);
        PendingIntent pi = PendingIntent.getBroadcast(c, (int) (due / 1000) ^ text.hashCode(), fire,
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        Intent open = c.getPackageManager().getLaunchIntentForPackage(c.getPackageName());
        PendingIntent show = PendingIntent.getActivity(c, 0, open, PendingIntent.FLAG_IMMUTABLE);
        am.setAlarmClock(new AlarmManager.AlarmClockInfo(due, show), pi);
        // Android 13+ needs a one-time yes before notifications can appear.
        if (Build.VERSION.SDK_INT >= 33
                && c.checkSelfPermission("android.permission.POST_NOTIFICATIONS") != PackageManager.PERMISSION_GRANTED) {
            activity.runOnUiThread(() -> activity.requestPermissions(new String[] {"android.permission.POST_NOTIFICATIONS"}, 77));
        }
        return "";
    }
}
