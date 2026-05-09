# Explosive Crypto Detector — Plan de Implementación

## Objetivo

Herramienta estática de análisis de código que detecta usos inseguros o vulnerables de criptografía en proyectos de software. Escanea repositorios o directorios en busca de patrones peligrosos: algoritmos deprecados, claves hardcodeadas, primitivas débiles, y configuraciones incorrectas.

---

## Fases

### Fase 1 — Estructura base del proyecto
- [ ] Inicializar proyecto Python (`pyproject.toml`, dependencias)
- [ ] Definir estructura de directorios:
  ```
  explosive_crypto_detector/
  ├── core/
  │   ├── scanner.py       # Motor principal de escaneo
  │   ├── rules.py         # Definición de reglas/patrones
  │   └── reporter.py      # Formateo de resultados
  ├── rules/
  │   └── *.yaml           # Reglas declarativas por categoría
  ├── cli.py               # Interfaz de línea de comandos
  └── tests/
  ```
- [ ] CLI básica con `argparse` o `typer`: `ecd scan <path>`

### Fase 2 — Motor de detección
- [ ] Implementar escáner basado en AST (Python `ast` module) para código Python
- [ ] Implementar escáner basado en regex para lenguajes adicionales (JS, Go, Java)
- [ ] Sistema de reglas en YAML: nombre, patrón, severidad, descripción, remediación
- [ ] Carga y validación de reglas en runtime

### Fase 3 — Reglas de detección

#### Categorías de riesgo:
| Categoría | Ejemplos |
|-----------|----------|
| Algoritmos deprecados | MD5, SHA1, DES, RC4, ECB mode |
| Claves y secretos hardcodeados | `secret_key = "abc123"`, tokens en código fuente |
| Números aleatorios inseguros | `random.random()` para criptografía |
| Configuraciones TLS débiles | SSLv2, SSLv3, TLS 1.0, `verify=False` |
| Longitud de clave insuficiente | RSA < 2048 bits, AES-128 en contextos de alta seguridad |
| Padding inseguro | PKCS#1 v1.5 para cifrado |

### Fase 4 — Reportes y salida
- [ ] Salida en consola con colores (severidad: CRITICAL / HIGH / MEDIUM / LOW)
- [ ] Exportar a JSON y SARIF (compatible con GitHub Code Scanning)
- [ ] Resumen al final del escaneo: total de hallazgos por severidad

### Fase 5 — Integración CI/CD
- [ ] GitHub Action: `explosive-crypto-detector-action`
- [ ] Código de salida no-cero si hay hallazgos CRITICAL o HIGH
- [ ] Modo `--fail-on <severity>` configurable

### Fase 6 — Tests y calidad
- [ ] Tests unitarios por regla con casos positivos y negativos
- [ ] Tests de integración sobre proyectos de ejemplo
- [ ] Cobertura mínima: 80%
- [ ] Linting: `ruff`, type checking: `mypy`

---

## Stack técnico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.11+ |
| CLI | Typer |
| Parsing AST | `ast` (stdlib) + `tree-sitter` para multi-lenguaje |
| Reglas | YAML + Pydantic para validación |
| Tests | pytest |
| CI | GitHub Actions |
| Empaquetado | `pyproject.toml` + PyPI |

---

## Criterios de éxito

- Detecta correctamente los patrones de la Fase 3 con < 5% de falsos positivos
- Escanea un proyecto de 50k LOC en menos de 10 segundos
- Output SARIF funciona con GitHub Code Scanning sin configuración adicional
- Instalable con `pip install explosive-crypto-detector`
