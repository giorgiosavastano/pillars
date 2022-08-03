import nox


@nox.session
def tests(session):
    session.install("pip", "numpy", "pytest", "scipy", "netCDF4")
    session.run("pip", "install", ".", "-v")
    session.run("pytest")
    # Here we queue up the test coverage session to run next
    session.notify("coverage")


@nox.session
def coverage(session):
    session.install("coverage")
    session.run("coverage")
