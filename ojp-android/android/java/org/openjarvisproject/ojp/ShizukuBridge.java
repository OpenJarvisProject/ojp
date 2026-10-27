package org.openjarvisproject.ojp;

import android.content.pm.PackageManager;

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
    private static final int REQUEST_CODE = 4242;
    private static final int MAX_OUTPUT = 64 * 1024;
    private static final long TIMEOUT_SECONDS = 30;

    private ShizukuBridge() {}

    /** "none" (not installed or not running), "old", "permission" (running, not allowed yet) or "ready". */
    static String state() {
        try {
            if (!Shizuku.pingBinder()) return "none";
            if (Shizuku.isPreV11()) return "old";
            return Shizuku.checkSelfPermission() == PackageManager.PERMISSION_GRANTED ? "ready" : "permission";
        } catch (Throwable t) {
            return "none";
        }
    }

    /** Shows Shizuku's "allow this app?" dialog. */
    static void requestPermission() {
        try {
            if (state().equals("permission")) Shizuku.requestPermission(REQUEST_CODE);
        } catch (Throwable ignored) {
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
        if (!p.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
            p.destroy();
            return "exit=124\n(timed out after " + TIMEOUT_SECONDS + " seconds)";
        }
        a.join(1000);
        b.join(1000);
        String text;
        synchronized (out) {
            text = new String(out.toByteArray(), StandardCharsets.UTF_8);
        }
        return "exit=" + p.exitValue() + "\n" + text;
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
