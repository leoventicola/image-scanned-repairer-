#!/usr/bin/env python3
"""Endereza páginas escaneadas, recorta y normaliza la tabla.

Uso:
  python preparar.py rotar
  python preparar.py recortar --debug
  python preparar.py ambos --padding 2
  python preparar.py normalizar

Flujo por carpetas (los nombres de archivo se conservan en cada paso):
  imagenes/ --rotar--> rotadas/ --recortar--> recortadas/
  recortadas/ --normalizar--> normalizadas/ (2070 × 3000 px)
  Los diagnósticos (--debug) se guardan en debug/.

Antes de recortar se valida la relación de aspecto contra RATIO_REF (±2%,
ajustable con --tolerancia; --sin-validar lo saltea). El gate es estricto:
el rectángulo conserva la tabla completa hasta el pie MODELO, incluida la
publicidad de pie si la hay.

Requiere: opencv-python, numpy. No utiliza OCR ni API.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np

EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.webp'}
DIR_IMAGENES = Path('imagenes')
DIR_ROTADAS = Path('rotadas')
DIR_RECORTADAS = Path('recortadas')
DIR_NORMALIZADAS = Path('normalizadas')
DIR_DEBUG = Path('debug')
ANCHO_NORMALIZADO = 2070
ALTO_NORMALIZADO = 3000
# Relación ancho/alto de referencia: mediana de las tablas aprobadas en
# validadas/ (0.6889-0.6919, desvío ±0.13%).
RATIO_REF = 0.690


def groups(indices):
    if len(indices) == 0:
        return []
    cuts = np.where(np.diff(indices) > 1)[0] + 1
    return [a for a in np.split(indices, cuts) if len(a)]


def binary_ink(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


def line_masks(gray):
    h, w = gray.shape
    ink = binary_ink(gray)
    # El escaneo corta las líneas en trazos discontinuos: cerrar los huecos
    # horizontales antes de abrir para recuperar los separadores completos.
    ink_h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 60), 1)))
    horizontal = cv2.morphologyEx(ink_h, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, w // 25), 1)))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 28))))
    return horizontal, vertical


def angulos_por_banda(gray):
    """Estima la inclinación que concentra la tinta en filas de cada banda."""
    h, w = gray.shape
    if w < 800:
        return []
    escala = 800 / w
    reducida = cv2.resize(gray, (800, max(1, round(h * escala))),
                          interpolation=cv2.INTER_AREA)
    tinta = binary_ink(reducida)
    h = tinta.shape[0]
    angulos = []
    for inicio, fin in ((.10, .35), (.35, .65), (.65, .93)):
        banda = tinta[round(inicio*h):round(fin*h), 96:720]
        alto, ancho = banda.shape
        if alto < 10 or not np.any(banda):
            continue
        margen = int(np.ceil(ancho * np.tan(np.deg2rad(5))))

        def concentracion(angulo):
            pendiente = -np.tan(np.deg2rad(angulo))
            matriz = np.array([[1, 0, 0], [pendiente, 1, margen]])
            alineada = cv2.warpAffine(banda, matriz, (ancho, alto + 2*margen))
            return alineada.sum(axis=1).var()

        grueso = max(np.linspace(-5, 5, 41), key=concentracion)
        finos = np.linspace(max(-5, grueso-.25), min(5, grueso+.25), 21)
        angulos.append(float(max(finos, key=concentracion)))
    return angulos


def detect_angle(gray):
    angulos = angulos_por_banda(gray)
    if len(angulos) == 3 and max(angulos) - min(angulos) <= .15:
        angulo = float(np.median(angulos))
        return angulo if abs(angulo) >= .08 else 0.0

    return detectar_angulo_hough(gray)


def detectar_angulo_hough(gray):
    h, w = gray.shape
    horizontal, _ = line_masks(gray)
    segments = cv2.HoughLinesP(horizontal, 1, np.pi / 1800,
                               threshold=max(50, w // 12),
                               minLineLength=max(80, w // 7), maxLineGap=25)
    angles, weights = [], []
    if segments is not None:
        for x1, y1, x2, y2 in segments[:, 0]:
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            length = np.hypot(x2-x1, y2-y1)
            if abs(angle) <= 5:
                angles.append(angle)
                weights.append(length)
    if not angles:
        return 0.0
    # Mediana robusta: las líneas de una publicidad no deben dominar.
    order = np.argsort(angles)
    sorted_angles = np.asarray(angles)[order]
    sorted_weights = np.asarray(weights)[order]
    angle = float(sorted_angles[np.searchsorted(np.cumsum(sorted_weights), sum(weights)/2)])
    return angle if abs(angle) >= 0.08 else 0.0


def rotate_keep_canvas(image, angle):
    if abs(angle) < 0.001:
        return image.copy()
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(np.ceil(h*sine + w*cosine))
    new_h = int(np.ceil(h*cosine + w*sine))
    matrix[0, 2] += (new_w-w)/2
    matrix[1, 2] += (new_h-h)/2
    return cv2.warpAffine(image, matrix, (new_w, new_h),
                          flags=cv2.INTER_CUBIC, borderValue=(255,255,255))


def strongest_peaks(profile, threshold, min_width=1):
    candidates = np.flatnonzero(profile >= threshold)
    return [(int(np.round(g.mean())), float(profile[g].max()))
            for g in groups(candidates) if len(g) >= min_width]


def horizontal_table_bounds(gray, h, w):
    """Límites de la grilla a partir de las filas horizontales anchas.

    El escaneo degrada las líneas en trazos discontinuos, así que se prueban
    cierres horizontales crecientes hasta recuperar suficientes separadores.
    Devuelve el rectángulo completo de la tabla (hasta el pie MODELO), incluyendo
    cualquier publicidad intercalada o de pie: el gate de aspecto exige que la
    tabla esté completa. Las bandas gruesas (>80 filas: publicidad o ruido de
    escaneo tenue) no aportan bordes laterales; el borde superior sale de la
    primera banda y el inferior de la última banda fina (pie MODELO).
    """
    ink = binary_ink(gray)
    for gap in (max(15, w // 60), max(30, w // 32), max(50, w // 20)):
        closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
        horizontal = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(35, w // 25), 1)))
        profile = (horizontal > 0).sum(axis=1)
        rows = np.flatnonzero(profile > w * .70)
        rows = rows[(rows > .015 * h) & (rows < .995 * h)]
        bands = groups(rows)
        thin = [b for b in bands if len(b) <= 80]
        if len(thin) < 20 or not bands:
            continue
        mins, maxs = [], []
        for band in thin:
            for y in band:
                xs = np.flatnonzero(horizontal[y] > 0)
                x0, x1 = int(xs.min()), int(xs.max())
                # Descartar filas puenteada hasta el borde de la hoja.
                if .025 * w < x0 and x1 < .985 * w:
                    mins.append(x0)
                    maxs.append(x1)
        if not mins:
            continue
        left = int(np.percentile(mins, 10))
        right = int(np.percentile(maxs, 90))
        top = int(bands[0].min())
        bottom = int(thin[-1].max())
        if right - left >= w * .60 and bottom - top >= h * .55:
            return left, top, right, bottom
    return None


def detect_bounds(gray):
    h, w = gray.shape
    horizontal, vertical = line_masks(gray)
    bounds = horizontal_table_bounds(gray, h, w)
    if bounds is not None:
        return bounds, horizontal, vertical
    # Camino alternativo: bordes verticales largos + separadores horizontales.
    # Buscar los bordes verticales largos, ignorando los bordes de la hoja.
    x_profile = (vertical > 0).sum(axis=0)
    x_peaks = strongest_peaks(x_profile, h * .36)
    x_peaks = [(x, score) for x, score in x_peaks if .025*w < x < .985*w]
    if len(x_peaks) < 2:
        raise ValueError('No se identificaron ambos bordes verticales de la tabla')
    # La primera y última línea larga son los límites de la grilla.
    left = min(x for x, _ in x_peaks)
    right = max(x for x, _ in x_peaks)
    if right-left < w*.70:
        if right - left < w * 0.6:
            print(
                "ADVERTENCIA: Bordes verticales incorrectos. "
                "Intentando detección alternativa..."
            )

            left, right = fallback_vertical_bounds(gray)

            if right - left < w * 0.6:
                raise ValueError(
                    "No se pudieron recuperar los bordes verticales"
                )

    # Buscar separadores horizontales largos dentro de los bordes.
    band = horizontal[:, left:right+1]
    y_profile = (band > 0).sum(axis=1)
    y_peaks = strongest_peaks(y_profile, (right-left)*.48)
    y_peaks = [(y, score) for y, score in y_peaks if .015*h < y < .995*h]
    if len(y_peaks) < 2:
        raise ValueError('No se encontraron límites horizontales de la tabla')
    top = min(y for y, _ in y_peaks)
    bottom = max(y for y, _ in y_peaks)
    # El borde derecho puede ser discontinuo; recuperarlo de la línea superior
    # completa, que incluye la última columna (2007).
    for yy in range(max(0, top-5), min(h, top+6)):
        xs = np.flatnonzero(horizontal[yy] > 0)
        if len(xs) > w * .55:
            candidate_left, candidate_right = int(xs.min()), int(xs.max())
            if .025*w < candidate_left < .15*w:
                left = min(left, candidate_left)
            if .85*w < candidate_right < .995*w:
                right = max(right, candidate_right)
    if bottom-top < h*.55:
        raise ValueError('La altura detectada es demasiado pequeña')
    return (left, top, right, bottom), horizontal, vertical


def fallback_vertical_bounds(gray):
    """
    Estima los bordes izquierdo y derecho utilizando
    las líneas horizontales de la tabla.
    """
    h, w = gray.shape

    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(50, w // 4), 1)
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    contours, _ = cv2.findContours(
        horizontal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)

        if cw >= w * 0.65:
            candidates.append((x, x + cw))

    if not candidates:
        raise ValueError(
            "No se pudieron estimar los bordes de la tabla"
        )

    left = int(np.median([x0 for x0, _ in candidates]))
    right = int(np.median([x1 for _, x1 in candidates]))

    return left, right


def listar_imagenes(carpeta):
    if not carpeta.is_dir():
        return []
    return sorted(x for x in carpeta.iterdir() if x.suffix.lower() in EXTS)


def rotar_imagen(path, outdir, angle_override=None):
    """Endereza una imagen y la guarda en outdir con el mismo nombre."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('No se pudo leer la imagen')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = detect_angle(gray) if angle_override is None else angle_override
    rotated = rotate_keep_canvas(image, angle)
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / path.name
    cv2.imwrite(str(target), rotated)
    print(f'{path.name}: rotación={angle:+.3f}° -> {target}')
    return target


def validar_relacion_aspecto(bounds, tolerancia):
    """Gate previo al recorte: la relación ancho/alto debe coincidir con la de
    las tablas aprobadas (validadas/). Estricto para todas las páginas."""
    left, top, right, bottom = bounds
    ratio = (right - left) / (bottom - top)
    desvio = abs(ratio - RATIO_REF) / RATIO_REF
    if desvio > tolerancia:
        raise ValueError(
            f'relación de aspecto fuera de rango: {ratio:.4f}, '
            f'esperado {RATIO_REF:.3f} ±{tolerancia:.0%} (desvío {desvio:.1%})'
        )
    return ratio


def recortar_tabla(path, outdir, debug=False, padding=2,
                   top_offset=0, bottom_offset=0, tolerancia=0.02, validar=True):
    """Recorta la tabla de una imagen ya enderezada, conservando el nombre."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('No se pudo leer la imagen')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bounds, _, _ = detect_bounds(gray)
    if validar:
        validar_relacion_aspecto(bounds, tolerancia)
    left, top, right, bottom = bounds
    h, w = gray.shape
    x0, y0 = max(0, left-padding), max(0, top-padding+top_offset)
    x1, y1 = min(w, right+padding+1), min(h, bottom+padding+1+bottom_offset)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Recorte vacío: revisá los offsets')
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / path.name
    cv2.imwrite(str(target), image[y0:y1, x0:x1])
    if debug:
        DIR_DEBUG.mkdir(exist_ok=True)
        marked = image.copy()
        cv2.rectangle(marked, (x0, y0), (x1-1, y1-1), (0, 0, 255), 3)
        cv2.imwrite(str(DIR_DEBUG / (path.stem + '_limites.png')), marked)
    print(f'{path.name}: límites={bounds}, recorte={x1-x0}x{y1-y0} -> {target}')


def normalizar_imagen(path, outdir):
    """Ajusta el tamaño del recorte y lo centra sobre un fondo blanco."""
    imagen = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if imagen is None:
        raise ValueError('No se pudo leer la imagen')
    alto, ancho = imagen.shape[:2]
    escala = min(ANCHO_NORMALIZADO / ancho, ALTO_NORMALIZADO / alto)
    ancho_ajustado = max(1, round(ancho * escala))
    alto_ajustado = max(1, round(alto * escala))
    interpolacion = cv2.INTER_AREA if escala <= 1 else cv2.INTER_CUBIC
    ajustada = cv2.resize(imagen, (ancho_ajustado, alto_ajustado),
                          interpolation=interpolacion)
    x = (ANCHO_NORMALIZADO - ancho_ajustado) // 2
    y = (ALTO_NORMALIZADO - alto_ajustado) // 2
    forma = (ALTO_NORMALIZADO, ANCHO_NORMALIZADO) + imagen.shape[2:]
    blanco = 1.0 if np.issubdtype(imagen.dtype, np.floating) else np.iinfo(imagen.dtype).max
    lienzo = np.full(forma, blanco, dtype=imagen.dtype)
    lienzo[y:y + alto_ajustado, x:x + ancho_ajustado] = ajustada
    outdir.mkdir(parents=True, exist_ok=True)
    destino = outdir / path.name
    if not cv2.imwrite(str(destino), lienzo):
        raise ValueError(f'No se pudo guardar la imagen en {destino}')
    print(f'{path.name}: normalizada a {ANCHO_NORMALIZADO}x{ALTO_NORMALIZADO} -> {destino}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('accion', choices=['rotar', 'recortar', 'ambos', 'normalizar'],
                   help='rotar: imagenes/ -> rotadas/; recortar: rotadas/ -> recortadas/; '
                        'ambos: rotar y recortar; normalizar: recortadas/ -> normalizadas/')
    p.add_argument('--debug', action='store_true', help='Guardar imagen con rectángulo de recorte en debug/')
    p.add_argument('--angle', type=float, help='Ángulo manual en grados (rotar/ambos); omitir para autodetección')
    p.add_argument('--padding', type=int, default=2, help='Margen alrededor de la tabla en píxeles')
    p.add_argument('--top-offset', type=int, default=0, help='Ajuste manual del borde superior en píxeles')
    p.add_argument('--bottom-offset', type=int, default=0, help='Ajuste manual del borde inferior en píxeles')
    p.add_argument('--tolerancia', type=float, default=0.02,
                   help='Tolerancia relativa de la relación de aspecto (default 0.02 = ±2%%)')
    p.add_argument('--sin-validar', action='store_true', help='Saltear la validación de relación de aspecto')
    args = p.parse_args()
    if args.angle is not None and args.accion == 'recortar':
        print('ADVERTENCIA: --angle se ignora en recortar; las imágenes de rotadas/ ya están enderezadas')
    errors = 0
    if args.accion == 'normalizar':
        paths = listar_imagenes(DIR_RECORTADAS)
        if not paths:
            p.error('No hay imágenes en recortadas/, ejecutá primero: python preparar.py recortar')
        for path in paths:
            try:
                normalizar_imagen(path, DIR_NORMALIZADAS)
            except Exception as exc:
                errors += 1
                print(f'ERROR {path.name}: {exc}')
    elif args.accion == 'recortar':
        paths = listar_imagenes(DIR_ROTADAS)
        if not paths:
            p.error('No hay imágenes en rotadas/, ejecutá primero: python preparar.py rotar')
        for path in paths:
            try:
                recortar_tabla(path, DIR_RECORTADAS, args.debug, args.padding,
                               args.top_offset, args.bottom_offset,
                               args.tolerancia, not args.sin_validar)
            except Exception as exc:
                errors += 1
                print(f'ERROR {path.name}: {exc}')
    else:
        paths = listar_imagenes(DIR_IMAGENES)
        if not paths:
            p.error(f'No se encontraron imágenes en {DIR_IMAGENES}/')
        for path in paths:
            try:
                rotada = rotar_imagen(path, DIR_ROTADAS, args.angle)
                if args.accion == 'ambos':
                    recortar_tabla(rotada, DIR_RECORTADAS, args.debug, args.padding,
                                   args.top_offset, args.bottom_offset,
                                   args.tolerancia, not args.sin_validar)
            except Exception as exc:
                errors += 1
                print(f'ERROR {path.name}: {exc}')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
