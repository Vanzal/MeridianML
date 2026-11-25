// Kein schwarzes Konsolenfenster unter Windows im Release-Build.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    meridian_desktop_lib::run()
}
