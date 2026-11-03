package org.openjarvisproject.ojp;

import android.app.Activity;
import android.app.AlarmManager;
import android.app.PendingIntent;
import android.app.Application;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.net.ConnectivityManager;
import android.net.LinkAddress;
import android.net.LinkProperties;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.os.Environment;
import android.os.StatFs;
import android.provider.AlarmClock;
import android.provider.ContactsContract;
import android.provider.Settings;
import android.app.Notification;
import android.content.ContentResolver;
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
        a.getApplication().registerActivityLifecycleCallbacks(new Application.ActivityLifecycleCallbacks() {
            public void onActivityResumed(Activity x) { visible = true; }
            public void onActivityPaused(Activity x) { visible = false; }
            public void onActivityCreated(Activity x, Bundle b) {}
            public void onActivityStarted(Activity x) {}
            public void onActivityStopped(Activity x) {}
            public void onActivitySaveInstanceState(Activity x, Bundle b) {}
            public void onActivityDestroyed(Activity x) {}
        });
        if (keepAlive(a)) startKeepAlive(a);
    }

    private static volatile boolean visible = true;

    private static SharedPreferences prefs(Context c) {
        return c.getSharedPreferences("ojp", Context.MODE_PRIVATE);
    }

    /** Whether the background service is switched on (it is, until the user turns it off). */
    static boolean keepAlive(Context c) {
        return prefs(c).getBoolean("keep_alive", true);
    }

    static void setKeepAlive(Context c, boolean on) {
        prefs(c).edit().putBoolean("keep_alive", on).apply();
    }

    private static void startKeepAlive(Context c) {
        try {
            c.startForegroundService(new Intent(c, KeepAliveService.class));
            askNotifications();
        } catch (Throwable t) {
            // Starting can be refused (battery limits); the app works the same without it.
        }
    }

    /** Android 13+ shows the service's notification, and the reply notifications, only after a one-time yes. */
    private static void askNotifications() {
        Context c = ctx();
        if (Build.VERSION.SDK_INT >= 33 && !prefs(c).getBoolean("asked_notifications", false)
                && c.checkSelfPermission("android.permission.POST_NOTIFICATIONS") != PackageManager.PERMISSION_GRANTED) {
            prefs(c).edit().putBoolean("asked_notifications", true).apply();
            activity.runOnUiThread(() -> activity.requestPermissions(new String[] {"android.permission.POST_NOTIFICATIONS"}, 77));
        }
    }

    /** Ask for a runtime permission and say so; the answer shows next time ("try again after allowing"). */
    private static boolean have(String permission, String what) throws Exception {
        if (Build.VERSION.SDK_INT < 23 || ctx().checkSelfPermission(permission) == PackageManager.PERMISSION_GRANTED) return true;
        activity.runOnUiThread(() -> activity.requestPermissions(new String[] {permission}, 78));
        throw new Exception("I need your permission to " + what + ". A question has appeared: allow it, then ask me again");
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
            case "keepalive_get":
                return keepAlive(ctx()) ? "on" : "off";
            case "keepalive_set": {
                boolean on = arg.equals("on");
                setKeepAlive(ctx(), on);
                if (on) startKeepAlive(ctx());
                else ctx().stopService(new Intent(ctx(), KeepAliveService.class));
                return "";
            }
            case "background":
                activity.runOnUiThread(() -> activity.moveTaskToBack(true));
                return "";
            case "busy":
                KeepAliveService.busy(ctx(), arg.equals("1"));
                return "";
            case "notify_reply":
                return notifyReply(arg);
            case "flashlight":
                return flashlight(arg.equals("on"));
            case "alarm":
                return alarm(arg);
            case "dial":
                return dial(arg);
            case "sms":
                return sms(arg);
            case "settings_page":
                return settingsPage(arg);
            case "brightness":
                return brightness(Integer.parseInt(arg.trim()));
            case "info":
                return info(arg);
            case "shizuku_state":
                return ShizukuBridge.state();
            case "shizuku_request":
                return ShizukuBridge.requestPermission();
            case "shizuku_manager":
                return ShizukuBridge.manager(ctx());
            case "open_shizuku": {
                // Whichever Shizuku build is installed (the original or a fork), found by the permission it declares.
                String found = ShizukuBridge.manager(ctx());
                if (found.isEmpty()) throw new Exception("No Shizuku app was found (the original and its forks both work)");
                Intent i = ctx().getPackageManager().getLaunchIntentForPackage(found.substring(found.indexOf('\t') + 1));
                if (i == null) throw new Exception("Couldn't open the Shizuku app");
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

    /** A notification when a reply is ready and the app isn't in front. */
    private static String notifyReply(String text) {
        if (visible || Build.VERSION.SDK_INT >= 33
                && ctx().checkSelfPermission("android.permission.POST_NOTIFICATIONS") != PackageManager.PERMISSION_GRANTED) return "";
        Context c = ctx();
        NotificationManager nm = (NotificationManager) c.getSystemService(Context.NOTIFICATION_SERVICE);
        nm.createNotificationChannel(new NotificationChannel("replies", "Replies", NotificationManager.IMPORTANCE_DEFAULT));
        Intent open = c.getPackageManager().getLaunchIntentForPackage(c.getPackageName());
        PendingIntent pi = PendingIntent.getActivity(c, 0, open, PendingIntent.FLAG_IMMUTABLE);
        String shown = text.length() > 300 ? text.substring(0, 300) + "…" : text;
        nm.notify(4712, new Notification.Builder(c, "replies").setSmallIcon(android.R.drawable.ic_dialog_email)
                .setContentTitle("OpenJarvis").setContentText(shown).setStyle(new Notification.BigTextStyle().bigText(shown))
                .setContentIntent(pi).setAutoCancel(true).build());
        return "";
    }

    private static String flashlight(boolean on) throws Exception {
        CameraManager cm = (CameraManager) ctx().getSystemService(Context.CAMERA_SERVICE);
        for (String id : cm.getCameraIdList()) {
            CameraCharacteristics cc = cm.getCameraCharacteristics(id);
            Boolean flash = cc.get(CameraCharacteristics.FLASH_INFO_AVAILABLE);
            Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
            if (Boolean.TRUE.equals(flash) && facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                cm.setTorchMode(id, on);
                return "";
            }
        }
        throw new Exception("this phone has no flashlight I can reach");
    }

    /** "hour TAB minute TAB label": set an alarm in the clock app, without opening it. */
    private static String alarm(String arg) throws Exception {
        String[] p = arg.split("\t", 3);
        Intent i = new Intent(AlarmClock.ACTION_SET_ALARM).putExtra(AlarmClock.EXTRA_HOUR, Integer.parseInt(p[0]))
                .putExtra(AlarmClock.EXTRA_MINUTES, Integer.parseInt(p[1])).putExtra(AlarmClock.EXTRA_SKIP_UI, true)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        if (p.length > 2 && !p[2].isEmpty()) i.putExtra(AlarmClock.EXTRA_MESSAGE, p[2]);
        if (i.resolveActivity(ctx().getPackageManager()) == null) throw new Exception("there is no clock app that can set alarms");
        activity.startActivity(i);
        return "";
    }

    /** A phone number as typed, or a contact's name looked up in the contacts. */
    private static String number(String who) throws Exception {
        String w = who.trim();
        if (w.matches("[+0-9()\\-. ]{3,}")) return w;
        have("android.permission.READ_CONTACTS", "look up your contacts");
        try (Cursor c = ctx().getContentResolver().query(ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
                new String[] {ContactsContract.CommonDataKinds.Phone.NUMBER, ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME},
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " LIKE ?", new String[] {"%" + w + "%"},
                ContactsContract.CommonDataKinds.Phone.DISPLAY_NAME + " COLLATE NOCASE ASC LIMIT 1")) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        }
        throw new Exception("I couldn't find " + w + " in your contacts");
    }

    /** Open the dialer with the number filled in; the user presses call. */
    private static String dial(String who) throws Exception {
        String n = number(who);
        activity.startActivity(new Intent(Intent.ACTION_DIAL, Uri.fromParts("tel", n, null)).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        return n;
    }

    /** "who TAB message": open the messages app with the text written; the user presses send. */
    private static String sms(String arg) throws Exception {
        String[] p = arg.split("\t", 2);
        String n = number(p[0]);
        activity.startActivity(new Intent(Intent.ACTION_SENDTO, Uri.fromParts("smsto", n, null))
                .putExtra("sms_body", p.length > 1 ? p[1] : "").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
        return n;
    }

    /** Open a page of the system settings (Android doesn't let apps flip most of these switches themselves). */
    private static String settingsPage(String page) throws Exception {
        String action;
        switch (page.trim()) {
            case "wifi": case "internet":
                action = Build.VERSION.SDK_INT >= 29 ? Settings.Panel.ACTION_INTERNET_CONNECTIVITY : Settings.ACTION_WIFI_SETTINGS; break;
            case "bluetooth": action = Settings.ACTION_BLUETOOTH_SETTINGS; break;
            case "airplane": action = Settings.ACTION_AIRPLANE_MODE_SETTINGS; break;
            case "display": case "brightness": action = Settings.ACTION_DISPLAY_SETTINGS; break;
            case "sound": case "volume": action = Settings.ACTION_SOUND_SETTINGS; break;
            case "battery": action = Intent.ACTION_POWER_USAGE_SUMMARY; break;
            case "storage": action = Settings.ACTION_INTERNAL_STORAGE_SETTINGS; break;
            case "apps": action = Settings.ACTION_APPLICATION_SETTINGS; break;
            case "location": action = Settings.ACTION_LOCATION_SOURCE_SETTINGS; break;
            case "nfc": action = Settings.ACTION_NFC_SETTINGS; break;
            case "date": action = Settings.ACTION_DATE_SETTINGS; break;
            case "language": action = Settings.ACTION_LOCALE_SETTINGS; break;
            case "accessibility": action = Settings.ACTION_ACCESSIBILITY_SETTINGS; break;
            case "notifications": action = Settings.ACTION_APP_NOTIFICATION_SETTINGS; break;
            case "about": action = Settings.ACTION_DEVICE_INFO_SETTINGS; break;
            case "developer": action = Settings.ACTION_APPLICATION_DEVELOPMENT_SETTINGS; break;
            default: action = Settings.ACTION_SETTINGS;
        }
        Intent i = new Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        if (action.equals(Settings.ACTION_APP_NOTIFICATION_SETTINGS)) i.putExtra(Settings.EXTRA_APP_PACKAGE, ctx().getPackageName());
        activity.startActivity(i);
        return page.trim();
    }

    /** Screen brightness 0-100 through the "modify system settings" permission the user can give once. */
    private static String brightness(int pct) throws Exception {
        Context c = ctx();
        if (!Settings.System.canWrite(c)) {
            activity.startActivity(new Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS, Uri.parse("package:" + c.getPackageName()))
                    .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK));
            throw new Exception("I need the \"modify system settings\" permission for that. The page is open: switch Open Jarvis on, then ask again");
        }
        Settings.System.putInt(c.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL);
        Settings.System.putInt(c.getContentResolver(), Settings.System.SCREEN_BRIGHTNESS, Math.max(1, Math.min(255, (int) Math.ceil(pct * 255 / 100.0))));
        return "";
    }

    /** Read-only facts about the phone that Android hands to any app: the no-Shizuku way to look things up. */
    private static String info(String arg) throws Exception {
        Context c = ctx();
        String[] p = arg.split("\t");
        switch (p[0]) {
            case "wifi": return wifi();
            case "ip": return ip();
            case "battery": return dispatch("battery", "");
            case "storage": {
                StatFs f = new StatFs(Environment.getExternalStorageDirectory().getPath());
                return "Phone storage: " + gb(f.getAvailableBytes()) + " free of " + gb(f.getTotalBytes());
            }
            case "version":
                return "Android " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + "), security patch " + Build.VERSION.SECURITY_PATCH;
            case "uptime": {
                long m = android.os.SystemClock.elapsedRealtime() / 60000;
                return "up " + m / 1440 + " days, " + m / 60 % 24 + " hours, " + m % 60 + " minutes";
            }
            case "model":
                return Build.MANUFACTURER + " " + Build.MODEL;
            case "setting": {
                // "setting TAB system|global|secure TAB key"
                ContentResolver r = c.getContentResolver();
                String v = p[1].equals("system") ? Settings.System.getString(r, p[2])
                        : p[1].equals("global") ? Settings.Global.getString(r, p[2]) : Settings.Secure.getString(r, p[2]);
                return v == null ? "null" : v;
            }
            case "packages": {
                PackageManager pm = c.getPackageManager();
                StringBuilder sb = new StringBuilder();
                for (ResolveInfo ri : pm.queryIntentActivities(new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER), 0)) {
                    if ((ri.activityInfo.applicationInfo.flags & android.content.pm.ApplicationInfo.FLAG_SYSTEM) != 0) continue;
                    sb.append("package:").append(ri.activityInfo.packageName).append(" (").append(ri.loadLabel(pm)).append(")\n");
                }
                return sb.toString();
            }
            default: throw new Exception("unknown topic " + p[0]);
        }
    }

    private static String gb(long bytes) {
        return String.format(Locale.ROOT, "%.1f GB", bytes / 1e9);
    }

    private static String ip() {
        ConnectivityManager cm = (ConnectivityManager) ctx().getSystemService(Context.CONNECTIVITY_SERVICE);
        Network n = cm.getActiveNetwork();
        LinkProperties lp = n == null ? null : cm.getLinkProperties(n);
        if (lp == null) return "not connected to a network";
        StringBuilder sb = new StringBuilder();
        for (LinkAddress a : lp.getLinkAddresses()) {
            if (a.getAddress() instanceof java.net.Inet4Address) sb.append(a.getAddress().getHostAddress()).append(" ");
        }
        return sb.length() == 0 ? "no IPv4 address" : "IP address " + sb.toString().trim() + " on " + lp.getInterfaceName();
    }

    @SuppressWarnings("deprecation")
    private static String wifi() throws Exception {
        Context c = ctx();
        WifiManager wm = (WifiManager) c.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
        if (!wm.isWifiEnabled()) return "Wi-Fi is off";
        WifiInfo w = wm.getConnectionInfo();
        if (w == null || w.getNetworkId() == -1) return "Wi-Fi is on but not connected to a network";
        String name = w.getSSID();
        if (name == null || name.contains("unknown ssid")) {
            // Android hides the network name from apps without the location permission.
            if (c.checkSelfPermission("android.permission.ACCESS_FINE_LOCATION") != PackageManager.PERMISSION_GRANTED) {
                activity.runOnUiThread(() -> activity.requestPermissions(new String[] {"android.permission.ACCESS_FINE_LOCATION"}, 79));
                name = "(name hidden by Android until you allow Location for Open Jarvis; a question just appeared, allow it and ask again)";
            } else {
                name = "(name hidden: switch Location on to see it)";
            }
        } else {
            name = name.replace("\"", "");
        }
        int mhz = w.getFrequency();
        String band = mhz >= 5925 ? "6 GHz" : mhz >= 4900 ? "5 GHz" : "2.4 GHz";
        return "Wi-Fi network " + name + ", " + band + " (" + mhz + " MHz), signal " + w.getRssi() + " dBm, link speed "
                + w.getLinkSpeed() + " Mbps, " + ip();
    }
}
