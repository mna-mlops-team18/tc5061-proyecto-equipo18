"""Pruebas de humo del sistema (operacion/verificar.py) contra un servidor HTTP falso.

``verificar.py`` es lo que corre el CI despues de ``docker compose up``; aqui se comprueba que
pasa con un sistema sano y que falla (codigo 1) si un endpoint falla o si no hay sistema, sin
necesitar Docker.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

import pytest

from operacion import verificar


class _ApiFalsa(BaseHTTPRequestHandler):
    """Responde como la API y como MLflow: lo minimo que revisa verificar.py."""

    rota = False  # con True, /predicciones/p2 responde 500

    def _responder(self, codigo: int, cuerpo) -> None:
        datos = (cuerpo if isinstance(cuerpo, str) else json.dumps(cuerpo)).encode()
        self.send_response(codigo)
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)

    def _get(self):
        modelos = [{"pregunta": p, "estado": "ok", "version": "0"} for p in ("p1", "p2", "p3")]
        respuestas = {
            "/salud": {"estado": "ok", "modelos": modelos},
            "/modelo": modelos,
            "/health": "OK",
        }
        if self.path in respuestas:
            self._responder(200, respuestas[self.path])
        else:
            self._responder(404, {})

    def _post(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if any(fila.get("hora_dia") == 30 for fila in cuerpo["instancias"]):
            self._responder(422, {"detail": "hora_dia"})
        elif self.rota and self.path.endswith("p2"):
            self._responder(500, {"detail": "roto"})
        else:
            self._responder(200, {"n": len(cuerpo["instancias"])})

    def log_message(self, *argumentos):  # sin ruido en la salida de pytest
        pass

    do_GET, do_POST = _get, _post  # nombres que exige http.server


@pytest.fixture
def servidor_falso():
    servidor = ThreadingHTTPServer(("127.0.0.1", 0), _ApiFalsa)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{servidor.server_address[1]}"
    servidor.shutdown()
    _ApiFalsa.rota = False


def test_verificar_pasa_con_un_sistema_sano(servidor_falso, capsys):
    assert verificar.main(["--api", servidor_falso, "--mlflow", servidor_falso]) == 0
    assert "FALLA" not in capsys.readouterr().out


def test_verificar_falla_si_un_endpoint_falla(servidor_falso, capsys):
    _ApiFalsa.rota = True
    assert verificar.main(["--api", servidor_falso, "--mlflow", servidor_falso]) == 1
    assert "| API /predicciones/p2 | FALLA |" in capsys.readouterr().out


def test_verificar_falla_si_no_hay_sistema():
    assert verificar.main(["--api", "http://127.0.0.1:9", "--mlflow", "http://127.0.0.1:9"]) == 1
