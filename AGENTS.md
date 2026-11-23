# AGENTS.md

## Proyecto
Recortador de tablas: endereza páginas escaneadas y recorta la tabla
de precios usando OpenCV (sin OCR ni APIs externas). Todo el código, comentarios
y mensajes de usuario están en español; mantener ese idioma.

## Entorno
- Python 3.12 con venv local: usar `.venv/bin/python` (no el python del sistema).
- Dependencias: `requirements_preparar.txt` (opencv-python >=4.8,<5; numpy >=1.24,<3).
- Instalar: `.venv/bin/pip install -r requirements_preparar.txt`
- No es repositorio git; no hay tests ni linter configurados.

## Ejecución
La acción es un argumento posicional; las carpetas son fijas (relativas al
directorio de trabajo) y los nombres de archivo se conservan en cada paso:

    .venv/bin/python preparar.py rotar      # imagenes/ -> rotadas/
    .venv/bin/python preparar.py recortar   # rotadas/ -> recortadas/
    .venv/bin/python preparar.py ambos --debug  # flujo completo

Opciones: `--angle` (ángulo manual, solo rotar/ambos; se ignora con advertencia
en recortar), `--padding`, `--top-offset`, `--bottom-offset`, `--debug`
(guarda `debug/<stem>_limites.png` con el rectángulo de recorte),
`--tolerancia` (tolerancia relativa del gate de aspecto, default 0.02 = ±2%)
y `--sin-validar` (saltea el gate).

## Validación de relación de aspecto
- `validar_relacion_aspecto` corre en `recortar_tabla` después de
  `detect_bounds` y antes de cortar/escribir: relación ancho/alto de los
  límites detectados contra `RATIO_REF = 0.690` (mediana de las tablas
  aprobadas en `validadas/`, rango 0.6889-0.6919).
- Gate estricto para todas las páginas, sin exenciones. Fuera de tolerancia:
  `ValueError`, no se escribe el recorte, `main()` reporta `ERROR` y continúa
  (código de salida 1).
- Para que el gate sea satisfacible, el recorte es el rectángulo completo de la
  tabla hasta el pie MODELO: la publicidad de pie o intercalada queda dentro
  del recorte (no se corta). Política decidida tras el caso 005, donde el corte
  de publicidad perdía la última fila y la exención del gate lo dejaba pasar.
- `validadas/` — recortes aprobados manualmente; fuente de `RATIO_REF`.

## Estructura de carpetas
- `imagenes/` — escaneos de entrada (no modificar ni borrar).
- `rotadas/` — imágenes enderezadas; salida de `rotar`, entrada de `recortar`.
- `recortadas/` — tablas recortadas (salida final).
- `debug/` — imágenes de diagnóstico con el rectángulo de límites.
- `validadas/` — recortes aprobados manualmente (referencia de aspecto).
- `tablas/` — salidas del flujo antiguo (legacy, no se usa ni se borra).
- `preparar.py` — script único (CLI + pipeline).

## Pipeline (orden de funciones)
- `rotar_imagen`: binary_ink → line_masks → detect_angle → rotate_keep_canvas
  → guarda en `rotadas/` con el mismo nombre.
- `recortar_tabla`: detect_bounds (con fallback_vertical_bounds) → recorte con
  padding/offsets → guarda en `recortadas/` con el mismo nombre.
- `ambos` procesa imagen por imagen: rotar → guardar → recortar lo guardado.

Detalles de detección:
- `detect_angle`: HoughLinesP sobre máscara de líneas horizontales; mediana
  ponderada por longitud, limitada a ±5°, ignora ángulos < 0.08°.
- `rotate_keep_canvas`: expande el lienzo para no perder contenido, fondo
  blanco, interpolación cúbica.
- `line_masks`: cierre horizontal previo (gap ~w/60) para puentear los trazos
  discontinuos del escaneo antes de la apertura morfológica.
- `detect_bounds`: intenta primero `horizontal_table_bounds`; si falla, usa el
  camino alternativo de bordes verticales + `fallback_vertical_bounds`.
- `horizontal_table_bounds`: prueba cierres horizontales crecientes
  (w/60, w/32, w/20) hasta recuperar ≥20 bandas finas de filas anchas (>70% del
  ancho). Rectángulo completo hasta el pie MODELO: las bandas gruesas (>80
  filas: publicidad o ruido de escaneo tenue) no aportan bordes laterales ni el
  inferior; la primera banda aporta el borde superior y la última banda fina el
  inferior. Bordes laterales: percentil 10 de los mínimos y 90 de los máximos
  por fila, descartando filas puenteadas hasta el borde de la hoja. Exige
  ancho ≥60% y alto ≥55% de la página.

## Convenciones
- Estilo funcional: funciones pequeñas, sin clases; snake_case.
- Errores por imagen: `raise ValueError` con mensaje en español; `main()` los
  captura, imprime `ERROR <archivo>: ...` y termina con código 1 si hubo fallos
  (pero continúa procesando el resto).
- `recortar` sin imágenes en `rotadas/` es un error de CLI: pide ejecutar
  `rotar` primero.
- Rutas con `pathlib.Path`; extensiones válidas en la constante `EXTS`
  (.png, .jpg, .jpeg, .tif, .tiff, .webp); carpetas en constantes `DIR_*`.
- Parámetros de detección son heurísticos (fracciones de w/h); al ajustarlos,
  verificar con `--debug` sobre varias imágenes de `imagenes/`.
