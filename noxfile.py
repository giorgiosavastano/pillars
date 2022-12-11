import nox


@nox.session
def tests(session):
    session.install("pip", "numpy", "pytest", "scipy")
    session.run("pip", "install", ".", "-v")
    session.run("pytest")
