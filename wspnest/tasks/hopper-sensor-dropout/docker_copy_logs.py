"""Harbor Docker backend for managed proxies with asynchronous noninteractive exec."""
import asyncio

from harbor.environments.base import ExecResult
from harbor.environments.docker.docker import DockerEnvironment


class DockerCopyLogs(DockerEnvironment):
    @property
    def capabilities(self):
        return super().capabilities.model_copy(update={"mounted": False})

    def _write_mounts_compose_file(self):
        return None

    async def _compose_exec(self, command, *, service, cwd, env, timeout_sec, user):
        container = await self._platform._resolve_service_container(service)
        args = ["exec", "-i"]
        if cwd:
            args.extend(["-w", cwd])
        if user is not None:
            args.extend(["-u", str(user)])
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        # Keep the Docker client's stdin attached without giving the task an input stream.
        args.extend([container, "bash", "-c", "exec bash -c \"$1\" </dev/null", "harbor", command])
        proc = await asyncio.create_subprocess_exec(
            *self._engine_cmd(*args),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            output = await asyncio.wait_for(proc.stdout.read(), timeout=timeout_sec)
            await proc.wait()
        except (asyncio.CancelledError, TimeoutError):
            await self._terminate_process(proc)
            raise
        finally:
            proc.stdin.close()
        stdout = output.decode(errors="replace")
        if proc.returncode and stdout:
            self.logger.debug("Attached exec exited %s: %s", proc.returncode, stdout)
        return ExecResult(stdout=stdout, return_code=proc.returncode)
