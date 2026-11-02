package org.openjarvisproject.ojp;

import android.content.Context;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PermissionInfo;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

import rikka.shizuku.Shizuku;

/**
 * Optional Shizuku support: with the Shizuku app running (started once over wireless debugging
 * or by root) and this app allowed in it, shell commands run with adb's rights. Without it every
 * call just reports that Shizuku isn't available; nothing else in the app depends on this.
 */
final class ShizukuBridge {
    /** The permission every Shizuku manager app (the original and its forks) declares for apps that want to use it. */
    private static final String API_PERMISSION = "moe.shizuku.manager.permission.API_V23";
    private static final int REQUEST_CODE = 4242;
    private static final int MAX_OUTPUT = 64 * 1024;
    private static final long TIMEOUT_SECONDS = 30;

    private ShizukuBridge() {}

    /**
     * Shizuku's API remembers its answers for as long as the process lives: a granted permission stays "granted" and
     * a "don't ask again" stays "denied", whatever the user later does inside Shizuku. After switching this app off
     * (or back to "ask") in Shizuku, the Settings button would then do nothing. Forget both remembered answers so the
     * next check asks Shizuku itself.
     */
    private static void forgetCachedAnswers() {
        for (String name : new String[] {"permissionGranted", "shouldShowRequestPermissionRationale"}) {
            try {
                java.lang.reflect.Field f = Shizuku.class.getDeclaredField(name);
                f.setAccessible(true);
                f.setBoolean(null, false);
            } catch (Throwable ignored) {
            }
        }
    }

    /**
     * "none" (not installed or not running), "old", "permission" (running, not allowed yet: the question can be
     * shown), "denied" (running, and Shizuku will not show the question again: it has to be switched on inside
     * Shizuku) or "ready".
     */
    static String state() {
        try {
            if (!Shizuku.pingBinder()) return "none";
            if (Shizuku.isPreV11()) return "old";
            forgetCachedAnswers();
            if (Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED) return "ready";
            return Shizuku.shouldShowRequestPermissionRationale() ? "denied" : "permission";
        } catch (Throwable t) {
            return "none";
        }
    }

    /**
     * The installed Shizuku manager app, whichever build it is: the package that declares Shizuku's API permission
     * (the original or any fork). Returns "label\tpackage", or "" when none is visible.
     */
    static String manager(Context ctx) {
        try {
            PackageManager pm = ctx.getPackageManager();
            for (PackageInfo pi : pm.getInstalledPackages(PackageManager.GET_PERMISSIONS)) {
                if (pi.permissions == null) continue;
                for (PermissionInfo perm : pi.permissions) {
                    if (API_PERMISSION.equals(perm.name)) {
                        CharSequence label = pi.applicationInfo != null ? pm.getApplicationLabel(pi.applicationInfo) : pi.packageName;
                        return label + "\t" + pi.packageName;
                    }
                }
            }
        } catch (Throwable ignored) {
        }
        return "";
    }

    /**
     * Shows Shizuku's "allow this app?" dialog. Returns "asked", or the state when the dialog cannot be shown
     * ("denied", "none", ...).
     */
    static String requestPermission() {
        try {
            String state = state();
            if (!state.equals("permission")) return state;
            Shizuku.requestPermission(REQUEST_CODE);
            return "asked";
        } catch (Throwable t) {
            return "none";
        }
    }

    /** Runs {@code sh -c command}; returns "exit=N" on the first line, then stdout and stderr. */
    static String run(String command) throws Exception {
        // newProcess is private in the API but is what it is meant to be used through for one-off commands.
        Method m = Shizuku.class.getDeclaredMethod("newProcess", String[].class, String[].class, String.class);
        m.setAccessible(true);
        Process p = (Process) m.invoke(null, new Object[] {new String[] {"sh", "-c", command}, null, null});
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Thread a = pump(p.getInputStream(), out);
        Thread b = pump(p.getErrorStream(), out);
        a.start();
        b.start();
        // Shizuku's remote process does not support the timed wait (it reported "hasn't exited" for a command that
        // had finished), so wait in a helper thread and give up after the timeout.
        final int[] code = {-1};
        final boolean[] done = {false};
        Thread waiter = new Thread(() -> {
            try {
                code[0] = p.waitFor();
                done[0] = true;
            } catch (Exception ignored) {
            }
        });
        waiter.start();
        waiter.join(TIMEOUT_SECONDS * 1000);
        if (!done[0]) {
            p.destroy();
            return "exit=124\n(timed out after " + TIMEOUT_SECONDS + " seconds)";
        }
        a.join(1000);
        b.join(1000);
        String text;
        synchronized (out) {
            text = new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
        return "exit=" + code[0] + "\n" + text;
    }

    private static Thread pump(InputStream in, ByteArrayOutputStream out) {
        return new Thread(() -> {
            byte[] buf = new byte[4096];
            try {
                int n;
                while ((n = in.read(buf)) > 0) {
                    synchronized (out) {
                        if (out.size() < MAX_OUTPUT) out.write(buf, 0, n);
                    }
                }
            } catch (Exception ignored) {
            }
        });
    }
}
