/**
 * U16Math — the pure arithmetic behind the uint16 viewer: a histogram of a
 * frame, percentiles off it, a lookup table from black, white and gamma, and
 * the fill of an RGBA buffer through that table. No DOM, so it is tested in
 * Node (tests/js/u16-math.test.mjs) and loaded as a plain script in the page.
 */
const U16Math = {
    /** Counts per value, 65536 bins. */
    histogram(u16) {
        const h = new Uint32Array(65536);
        for (let i = 0; i < u16.length; i++) h[u16[i]]++;
        return h;
    },

    /**
     * The values at the given fractions (0..1) of the pixels, from a histogram.
     * percentiles(h, n, [0.01, 0.995]) → [black, white] for an auto contrast.
     */
    percentiles(hist, total, fractions) {
        const out = new Array(fractions.length).fill(0);
        if (!total) return out;
        const targets = fractions.map(f => Math.max(0, Math.min(total, Math.round(f * total))));
        let acc = 0, next = 0;
        for (let v = 0; v < hist.length && next < targets.length; v++) {
            acc += hist[v];
            while (next < targets.length && acc >= targets[next]) { out[next] = v; next++; }
        }
        return out;
    },

    /**
     * A table from value to display grey: 0 at or below black, 255 at or above
     * white, a gamma curve between. gamma 1 is linear; below 1 lifts the dark end.
     */
    lut(black, white, gamma = 1) {
        const t = new Uint8ClampedArray(65536);
        const lo = Math.max(0, Math.min(65535, black | 0));
        const hi = Math.max(lo + 1, Math.min(65535, white | 0));
        const span = hi - lo, g = gamma > 0 ? gamma : 1;   // pow(x, γ): γ below 1 lifts the dark end
        for (let v = 0; v < 65536; v++) {
            if (v <= lo) { t[v] = 0; continue; }
            if (v >= hi) { t[v] = 255; continue; }
            t[v] = Math.round(255 * Math.pow((v - lo) / span, g));
        }
        return t;
    },

    /** Fill an RGBA byte buffer (4 per pixel) from the frame through the table. */
    shade(u16, lut, rgba) {
        for (let i = 0, j = 0; i < u16.length; i++, j += 4) {
            const g = lut[u16[i]];
            rgba[j] = g; rgba[j + 1] = g; rgba[j + 2] = g; rgba[j + 3] = 255;
        }
        return rgba;
    },

    /** The frame's shape and range from the response headers gently.core.u16 writes. */
    parseHeaders(get) {
        const n = k => { const v = get(`X-Image-${k}`); return v == null || v === '' ? null : Number(v); };
        return {
            width: n('Width'), height: n('Height'), scale: n('Scale') || 1,
            sourceWidth: n('Source-Width'), sourceHeight: n('Source-Height'),
            min: n('Min'), max: n('Max'), planes: n('Planes'), plane: n('Plane'),
        };
    },
};

if (typeof module !== 'undefined' && module.exports) module.exports = U16Math;
