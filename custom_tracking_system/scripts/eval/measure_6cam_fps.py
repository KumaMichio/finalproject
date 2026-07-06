"""measure_6cam_fps.py — Do throughput (FPS tong) cua pipeline khi chay N camera
dong thoi, doi chieu voi muc tieu ~15-20 FPS (Chuong 6.2.2).

Cach do: pipeline chay 1 main loop xu ly DONG BO tat ca camera moi vong lap, nen
`GET /api/stats/ -> fps` chinh la FPS tong (so vong lap/giay tren toan bo N cam).
Script chi POLL endpoint do — KHONG tu khoi dong backend.

QUY TRINH DO (tren may GPU demo, khong phai may dev CPU):

  1) Khoi dong backend voi 6 luong. Cach don gian nhat: lap video Pho Hue x6
     lam 6 nguon file (thay throughput o dung do phan giai that 1600x1200):

       cd server
       ..\custom_tracking_system\venv_tracking\Scripts\python.exe app.py --with-ai ^
         --source file ^
         --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" ^
         --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" ^
         --video-path "<PHO_HUE>.mp4" --video-path "<PHO_HUE>.mp4" ^
         --camera-ids CAM1,CAM2,CAM3,CAM4,CAM5,CAM6 ^
         --config ..\custom_tracking_system\config\camera_config_pho_hue.yaml ^
         --loop-video

     (Route Predictor chi gan camera_0; 5 cam con lai van do detect+track+
     violation — du de do tai throughput 6 luong.)

  2) Doi ~15s cho pipeline nap model + on dinh, roi chay script nay:

       python measure_6cam_fps.py --duration 60 --warmup 15

Ket qua in ra: mean / median / p10 / min / max FPS + so camera active + PASS/FAIL
so voi khoang muc tieu. Xuat kem JSON de dua vao bao cao/checklist.
"""
import argparse
import json
import time
import urllib.request
import urllib.error


def poll_stats(url: str):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def percentile(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description="Do FPS tong pipeline N-camera")
    ap.add_argument("--url", default="http://localhost:8000/api/stats/")
    ap.add_argument("--duration", type=float, default=60.0, help="Giay do")
    ap.add_argument("--warmup", type=float, default=15.0,
                    help="Giay bo qua dau (cho model nap + on dinh)")
    ap.add_argument("--interval", type=float, default=1.0, help="Chu ky poll (s)")
    ap.add_argument("--target-min", type=float, default=15.0)
    ap.add_argument("--target-max", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="Ghi ket qua JSON (tuy chon)")
    args = ap.parse_args()

    # Sanity: endpoint song?
    try:
        first = poll_stats(args.url)
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(f"[LOI] Khong goi duoc {args.url} ({e}). "
                         f"Backend da chay --with-ai chua?")

    n_cam = first.get("active_cameras", 0)
    print(f"Backend OK. active_cameras={n_cam}, source={first.get('source')}, "
          f"map_enabled={first.get('map_enabled')}")
    if args.warmup > 0:
        print(f"Warmup {args.warmup:.0f}s (bo qua so lieu nap model)...")
        time.sleep(args.warmup)

    samples, cams = [], []
    t_end = time.perf_counter() + args.duration
    print(f"Do trong {args.duration:.0f}s (poll moi {args.interval:.1f}s)...")
    while time.perf_counter() < t_end:
        try:
            s = poll_stats(args.url)
            fps = float(s.get("fps", 0) or 0)
            samples.append(fps)
            cams.append(int(s.get("active_cameras", 0) or 0))
            print(f"  fps={fps:6.2f}  active_cameras={cams[-1]}")
        except (urllib.error.URLError, OSError, ValueError) as e:
            print(f"  [bo qua] poll loi: {e}")
        time.sleep(args.interval)

    # Bo cac mau fps==0 (chua co frame) khoi thong ke
    valid = [x for x in samples if x > 0]
    if not valid:
        raise SystemExit("[LOI] Khong co mau FPS > 0. Pipeline chua xu ly frame?")

    sv = sorted(valid)
    res = {
        "n_samples": len(valid),
        "active_cameras": max(cams) if cams else 0,
        "fps_mean": round(sum(sv) / len(sv), 2),
        "fps_median": round(percentile(sv, 0.5), 2),
        "fps_p10": round(percentile(sv, 0.10), 2),
        "fps_min": round(sv[0], 2),
        "fps_max": round(sv[-1], 2),
        "target_min": args.target_min,
        "target_max": args.target_max,
    }
    res["meets_target"] = res["fps_median"] >= args.target_min

    print("\n" + "=" * 48)
    print(f"  So camera        : {res['active_cameras']}")
    print(f"  FPS mean/median  : {res['fps_mean']} / {res['fps_median']}")
    print(f"  FPS p10/min/max  : {res['fps_p10']} / {res['fps_min']} / {res['fps_max']}")
    print(f"  Muc tieu         : {args.target_min}-{args.target_max} FPS")
    print(f"  KET LUAN         : {'PASS' if res['meets_target'] else 'CHUA DAT'} "
          f"(median {res['fps_median']} vs muc tieu >= {args.target_min})")
    print("=" * 48)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"Da ghi {args.out}")


if __name__ == "__main__":
    main()
