
import os
import math
import time
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from flask import Flask, render_template, jsonify, request

try:  
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np

try:
    from sklearn.linear_model import Ridge
except ImportError:  
    Ridge = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.json.ensure_ascii = False


W = (0.25, 0.40, 0.15, 0.20)          
KRITIS, WASPADA, WTD_KRITIS = 0.7, 0.4, -60
BASE = (-2.31, 114.01)               
CANAL = {"A": (-2.2985, 113.9985), "B": (-2.2985, 114.0185), "C": (-2.3185, 113.9985), "D": (-2.3185, 114.0185)}
SLOPE = {"A": 0.003, "B": 0.010, "C": 0.018, "D": 0.008}   
BLOK = "ABCD"

SENSOR = [("S1", "A", -2.296, 113.996, 62, -20, 27.5, 2.8), ("S2", "A", -2.304, 114.004, 58, -28, 27.8, 3.0),
          ("S3", "B", -2.296, 114.016, 50, -38, 29.0, 4.0), ("S4", "B", -2.304, 114.024, 47, -42, 29.5, 4.2),
          ("S5", "C", -2.316, 113.996, 34, -58, 30.5, 4.8), ("S6", "C", -2.324, 114.004, 30, -66, 31.0, 5.0),
          ("S7", "D", -2.316, 114.016, 44, -46, 29.8, 4.4), ("S8", "D", -2.324, 114.024, 40, -50, 30.2, 4.6)]
 
WILAYAH = [
    ("pulang-pisau", "Kabupaten Pulang Pisau", "Kalimantan Tengah", -2.745, 114.260, (-6, -10, 1.0, 0.7)),
    ("kapuas", "Kabupaten Kapuas", "Kalimantan Tengah", -3.000, 114.390, (-3, -6, 0.6, 0.4)),
    ("kotim", "Kabupaten Kotawaringin Timur", "Kalimantan Tengah", -2.530, 112.950, (-3, -5, 0.5, 0.3)),
    ("kobar", "Kabupaten Kotawaringin Barat", "Kalimantan Tengah", -2.680, 111.620, (0, -1, 0.1, 0.0)),
    ("seruyan", "Kabupaten Seruyan", "Kalimantan Tengah", -3.200, 112.500, (1, 1, 0.0, 0.0)),
    ("katingan", "Kabupaten Katingan", "Kalimantan Tengah", -1.890, 113.400, (-2, -3, 0.3, 0.2)),
    ("sukamara", "Kabupaten Sukamara", "Kalimantan Tengah", -2.630, 111.220, (5, 7, -0.5, -0.3)),
    ("barsel", "Kabupaten Barito Selatan", "Kalimantan Tengah", -1.730, 114.830, (7, 9, -0.7, -0.5)),
    ("bartim", "Kabupaten Barito Timur", "Kalimantan Tengah", -1.800, 115.150, (9, 12, -1.0, -0.7)),
    ("oki", "Kabupaten Ogan Komering Ilir", "Sumatera Selatan", -3.450, 105.100, (-5, -9, 0.9, 0.6)),
    ("ketapang", "Kabupaten Ketapang", "Kalimantan Barat", -1.900, 110.100, (-2, -4, 0.4, 0.2)),
    ("sintang", "Kabupaten Sintang", "Kalimantan Barat", 0.070, 111.490, (12, 16, -1.5, -1.0)),
    ("batola", "Kabupaten Barito Kuala", "Kalimantan Selatan", -2.980, 114.700, (3, 4, -0.2, -0.1)),
    ("tapin", "Kabupaten Tapin", "Kalimantan Selatan", -2.940, 115.000, (4, 5, -0.4, -0.2)),
]
REG = {w[0]: dict(id=w[0], nama=w[1], prov=w[2], lat=w[3], lon=w[4], off=w[5], idx=i) for i, w in enumerate(WILAYAH)}
DEFAULT_WILAYAH = "pulang-pisau"


def get_reg(wid):
    return REG.get(wid) or REG[DEFAULT_WILAYAH]


def geser(reg, la, lo):
    
    return round(reg["lat"] + (la - BASE[0]), 5), round(reg["lon"] + (lo - BASE[1]), 5)


SKN = {"normal": (0, 0, 0, 0), "kemarau": (-12, -14, 2.5, 1.5), "hujan": (15, 18, -2, -1.5)}
DRIFT = {"normal": 0.0, "kemarau": 0.012, "hujan": -0.025}
ARAH8 = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"]

LOG = []
LOG_LOCK = threading.Lock()
LOG_MAX = 500

clip = lambda v, a=0.0, b=1.0: max(a, min(b, v))


def num(val, default, lo=None, hi=None):
    
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def tambah_log(jenis, isi, tgl=None, wid=None):
    reg = get_reg(wid) if wid else None
    with LOG_LOCK:
        LOG.append(dict(tgl=tgl or datetime.now().strftime("%Y-%m-%d"), jenis=jenis, isi=isi,
                        wid=reg["id"] if reg else "", wilayah=reg["nama"] if reg else "Semua wilayah"))
        del LOG[:-LOG_MAX]



def pfvi(sm, wtd, ts, ev):
    """PFVI = w1*Sm + w2*Wtd + w3*Ts + w4*Ev, tiap variabel dinormalisasi 0-1 (1 = paling kering/rentan)."""
    n = (clip((70 - sm) / 50), clip(-wtd / abs(WTD_KRITIS)), clip((ts - 26) / 12), clip(ev / 8))
    return clip(sum(w * x for w, x in zip(W, n)))


def status(p):
    return "Kritis" if p > KRITIS else "Waspada" if p >= WASPADA else "Aman"


def nama_arah(d):
    return ARAH8[int(round((d % 360) / 45)) % 8]


def angin(kmh, dari):
    """Vektor angin 2D. dari = arah ASAL angin (derajat, 0=Utara). Risiko merambat ke arah (dari+180)."""
    th, ms = math.radians(dari), kmh / 3.6
    vx, vy = -ms * math.sin(th), -ms * math.cos(th)         
    tuju = (dari + 180) % 360
    return dict(kmh=kmh, dari=dari, dari_nama=nama_arah(dari), tuju=tuju, tuju_nama=nama_arah(tuju),
                vx=round(vx, 2), vy=round(vy, 2))


def sensor_base(reg):
    
    o = reg["off"]
    out = []
    for sid, b, la, lo, sm, wtd, ts, ev in SENSOR:
        lat, lon = geser(reg, la, lo)
        out.append((sid, b, lat, lon, clip(sm + o[0], 5, 95), min(-2, wtd + o[1]), ts + o[2], max(0.3, ev + o[3])))
    return out


def sensor_now(skn, reg):
    t, d = time.time(), SKN[skn]
    out = []
    for i, (sid, b, lat, lon, sm, wtd, ts, ev) in enumerate(sensor_base(reg)):
        j = math.sin(t / 15 + i + reg["idx"])
        v = dict(sm=clip(sm + d[0] + 1.2 * j, 5, 95), wtd=min(-2, wtd + d[1] + 1.5 * j),
                 ts=ts + d[2] + 0.3 * j, ev=max(0.3, ev + d[3] + 0.2 * j))
        p = pfvi(v["sm"], v["wtd"], v["ts"], v["ev"])
        out.append(dict(id=sid, blok=b, lat=lat, lon=lon, **{k: round(x, 1) for k, x in v.items()},
                        pfvi=round(p, 2), status=status(p)))
    return out


def idw_grid(sens, wind, reg, n=14):
    """Interpolasi spasial IDW: D = sqrt((x-xi)^2+(y-yi)^2), risiko = sum(PFVI_i/D^2)/sum(1/D^2)
    + dorongan angin dari sensor kritis. Grid berpusat di wilayah terpilih."""
    la0, la1, lo0, lo1 = reg["lat"] - 0.025, reg["lat"] + 0.025, reg["lon"] - 0.025, reg["lon"] + 0.025
    br = math.radians(wind["tuju"])
    ux, uy = math.sin(br), math.cos(br)
    cells = []
    for r in range(n):
        row = []
        for c in range(n):
            y, x = la1 - (r + .5) * (la1 - la0) / n, lo0 + (c + .5) * (lo1 - lo0) / n
            numr = den = boost = 0.0
            for s in sens:
                dx, dy = (x - s["lon"]) * 111000, (y - s["lat"]) * 111000
                D = math.hypot(dx, dy) + 1
                w = 1 / D ** 2
                numr += w * s["pfvi"]
                den += w
                if s["pfvi"] > KRITIS and D < 2500:
                    cosang = (dx * ux + dy * uy) / D
                    boost += 0.18 * s["pfvi"] * max(0, cosang) * math.exp(-D / 1200) * min(1, wind["kmh"] / 15)
            row.append(round(clip(numr / den + boost), 2))
        cells.append(row)
    return dict(la1=la1, lo0=lo0, dla=(la1 - la0) / n, dlo=(lo1 - lo0) / n, cells=cells)


def history(b, p0, reg):
    rng = np.random.default_rng(ord(b) * 100 + reg["idx"])
    return [round(clip(p0 - SLOPE[b] * (6 - i) + rng.normal(0, .008)), 3) for i in range(7)]


def forecast(hist, now, skn):
    X, y = np.arange(7).reshape(-1, 1), np.array(hist)
    if Ridge:
        m = Ridge(alpha=0.1).fit(X, y)
        f = lambda xs: m.predict(np.array(xs).reshape(-1, 1))
    else:
        a, c = np.polyfit(np.arange(7), y, 1)
        f = lambda xs: a * np.array(xs) + c
    base = f(list(range(7, 21))) - f([6])[0] + now
    return [round(clip(float(v) + DRIFT[skn] * (k + 1)), 3) for k, v in enumerate(base)]


def blok_data(sens, skn, wind, reg):
    res = {}
    base = sensor_base(reg)
    for b in BLOK:
        ss = [s for s in sens if s["blok"] == b]
        mean = lambda k: sum(s[k] for s in ss) / len(ss)
        now = round(sum(s["pfvi"] for s in ss) / len(ss), 2)
        bs = [x for x in base if x[1] == b]
        base_p = round(sum(pfvi(*x[4:8]) for x in bs) / len(bs), 3)
        h = history(b, base_p, reg)
        fc = forecast(h, now, skn)
        lead = next((i + 1 for i, v in enumerate(fc) if v > KRITIS), None)
        st = status(now)
        wtd = round(mean("wtd"))
        R = round(1.5 * math.exp(0.08 * wind["vx"] + 0.08 * wind["vy"]) * now ** 2 * 10, 1)  
        if st == "Kritis":
            kal = f"Peringatan: Blok {b} sudah di Zona Kritis (PFVI {now:.2f}, muka air {wtd} cm)"
        elif lead:
            kal = f"Peringatan Dini: Blok {b} memasuki Zona Kritis dalam {lead} hari ke depan"
        else:
            kal = f"Kondisi Lahan Blok {b}: {st} (Indeks Kerentanan {'Rendah' if st == 'Aman' else 'Sedang'})"
        rec = None
        if st == "Kritis" or lead:
            peak = max(fc + [now])
            tgl = datetime.now() + timedelta(days=1 if (st == "Kritis" or lead <= 3) else lead - 2)
            lat, lon = geser(reg, *CANAL[b])
            dur = 4 + round(8 * peak)
            rec = dict(tanggal=tgl.strftime("%d %b %Y"), jam="08.00 WIB", durasi_jam=dur, lat=lat, lon=lon,
                       sektor=f"Sektor {1 if b in 'AB' else 2} Blok {b}")
        res[b] = dict(blok=b, pfvi=now, status=st, wtd=wtd, sm=round(mean("sm")), ts=round(mean("ts"), 1),
                      ev=round(mean("ev"), 1), hist=h, fc=fc, lead=lead, kalimat=kal, R=R, rec=rec,
                      pesan=pesan_wa(b, now, st, wtd, wind, rec, lead, reg))
    return res


def ringkas_wilayah(skn, wind):
    """Ringkasan semua wilayah: status dari rata-rata PFVI empat blok, jumlah blok kritis, jeda tercepat menuju kritis."""
    out = []
    for reg in REG.values():
        bl = blok_data(sensor_now(skn, reg), skn, wind, reg)
        mean = round(sum(x["pfvi"] for x in bl.values()) / len(bl), 2)
        leads = [x["lead"] for x in bl.values() if x["lead"] and x["status"] != "Kritis"]
        out.append(dict(id=reg["id"], nama=reg["nama"], prov=reg["prov"], lat=reg["lat"], lon=reg["lon"],
                        pfvi=mean, status=status(mean), kritis=sum(1 for x in bl.values() if x["status"] == "Kritis"),
                        lead=min(leads) if leads else None))
    return out


def pesan_wa(b, p, st, wtd, wind, rec, lead, reg):
    t = (f"*PERINGATAN DINI AI-PFVI*\n{reg['nama']}, {reg['prov']}\n"
         f"Blok {b}: PFVI {p:.2f} ({st.upper()})\nMuka air {wtd} cm (batas kritis -60 cm)\n")
    t += f"Angin dari {wind['dari_nama']} {wind['kmh']:.0f} km/jam, risiko merambat ke {wind['tuju_nama']}\n"
    if lead:
        t += f"Perkiraan masuk zona kritis: {lead} hari lagi\n"
    if rec:
        t += (f"Tindakan: pembasahan kanal {rec['sektor']} di titik {rec['lat']:.4f}, {rec['lon']:.4f} "
              f"pada {rec['tanggal']} pukul {rec['jam']} selama {rec['durasi_jam']} jam.")
    else:
        t += "Tindakan: pantau rutin, belum perlu pembasahan."
    return t


def seed_log():
    n = datetime.now()
    for reg in REG.values():
        k = reg["idx"]
        for d, jenis, tx in [(5 + k % 3, "anomali", "S6 Blok C: muka air turun 9 cm dalam 24 jam"),
                             (4, "tindakan", "Kelompok tani membasahi kanal Sektor 2 selama 6 jam"),
                             (2, "anomali", "S4 Blok B: kelembapan tanah di bawah 48%"),
                             (1, "tindakan", "Penutupan sekat kanal Blok B oleh kelompok tani")]:
            tambah_log(jenis, tx, (n - timedelta(days=d)).strftime("%Y-%m-%d"), reg["id"])


seed_log()



def normalisasi_nomor(raw):
    """Ubah berbagai format nomor Indonesia menjadi 628xxxxxxxxxx.
    Terima: 081234567890, +62 812-3456-7890, 6281234567890, 81234567890, 0062812..., +62 0812..."""
    no = "".join(ch for ch in str(raw or "") if ch.isdigit())
    if no.startswith("00"):
        no = no[2:]
    if no.startswith("620"):
        no = "62" + no[3:]
    elif no.startswith("0"):
        no = "62" + no[1:]
    elif no.startswith("8"):
        no = "62" + no
    return no if 10 <= len(no) <= 15 else ""


def link_wa(no, msg):
    return f"https://wa.me/{no}?text={urllib.parse.quote(msg)}"


def kirim_fonnte(token, no, msg):
    """Kirim lewat gateway Fonnte. Return (ok, detail)."""
    data = urllib.parse.urlencode({"target": no, "message": msg, "countryCode": "62"}).encode()
    req = urllib.request.Request("https://api.fonnte.com/send", data=data, headers={"Authorization": token, "User-Agent": "Mozilla/5.0 AI-PFVI"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
    try:
        r = json.loads(body)
    except ValueError:
        return False, "Balasan gateway tidak dapat dibaca"
    ok = bool(r.get("status"))
    return ok, r.get("reason") or r.get("detail") or ("terkirim" if ok else "gagal")


@app.route("/sw.js")
def sw():
    resp = app.send_static_file("sw.js")
    resp.headers["Content-Type"] = "application/javascript"
    resp.headers["Service-Worker-Allowed"] = "/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route("/")
def index():
    return render_template("index.html")


def parse_args():
    skn = request.args.get("skenario", "normal")
    skn = skn if skn in SKN else "normal"
    kmh = num(request.args.get("kmh"), 14, 0, 80)
    dari = num(request.args.get("dari"), 135) % 360
    return skn, angin(kmh, dari), get_reg(request.args.get("wilayah"))


@app.route("/api/data")
def api_data():
    skn, w, reg = parse_args()
    sens = sensor_now(skn, reg)
    bl = blok_data(sens, skn, w, reg)
    return jsonify(wilayah=dict(id=reg["id"], nama=reg["nama"], prov=reg["prov"], lat=reg["lat"], lon=reg["lon"]),
                   sensor=sens, blok=bl, angin=w, grid=idw_grid(sens, w, reg),
                   waktu=datetime.now().strftime("%H:%M:%S"),
                   mode_wa="gateway" if os.getenv("FONNTE_TOKEN") else "link")


@app.route("/api/wilayah")
def api_wilayah():
    skn, w, _ = parse_args()
    return jsonify(wilayah=ringkas_wilayah(skn, w), waktu=datetime.now().strftime("%H:%M:%S"))


@app.route("/api/log")
def api_log():
    with LOG_LOCK:
        return jsonify(list(LOG))


@app.route("/api/health")
def health():
    return jsonify(ok=True, waktu=datetime.now().isoformat(timespec="seconds"),
                   mode_wa="gateway" if os.getenv("FONNTE_TOKEN") else "link")


@app.route("/api/wa/kirim", methods=["POST"])
def wa_kirim():
    j = request.get_json(silent=True) or {}
    no = normalisasi_nomor(j.get("nomor"))
    msg = str(j.get("pesan") or "").strip()[:4000]
    blok = str(j.get("blok") or "-").upper()
    blok = blok if blok in BLOK else "-"
    wid = get_reg(j.get("wilayah"))["id"]
    if not no or not msg:
        return jsonify(ok=False, error="Nomor atau pesan tidak valid. Contoh nomor: 081234567890"), 400

    url = link_wa(no, msg)
    token = (os.getenv("FONNTE_TOKEN") or "").strip()
    if token:
        try:
            ok, detail = kirim_fonnte(token, no, msg)
        except Exception as e: 
            return jsonify(ok=False, mode="gateway", url=url, error=f"Gateway gagal: {e}"), 502
        if ok:
            tambah_log("notifikasi", f"WhatsApp terkirim ke {no[:5]}***: Blok {blok}", wid=wid)
            return jsonify(ok=True, mode="gateway", detail=detail)
        return jsonify(ok=False, mode="gateway", url=url, error=f"Gateway menolak: {detail}"), 502

    tambah_log("notifikasi", f"WhatsApp disiapkan untuk {no[:5]}***: Blok {blok}", wid=wid)
    return jsonify(ok=True, mode="link", url=url)


if __name__ == "__main__":
    
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG") == "1")
