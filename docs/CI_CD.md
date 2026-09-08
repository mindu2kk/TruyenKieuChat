# CI/CD

The `CI/CD` workflow runs this sequence:

`Lint -> Test -> Coverage -> Django production readiness -> Vercel preview -> Production migration -> Production deploy -> Health check -> Rollback`.

Pull requests run verification only. A push to `main` or `master` first deploys
a Vercel Preview to the GitHub `staging` environment. The `production`
environment then runs migrations, deploys the prebuilt Vercel artifact and
checks `/api/health/`. If production health fails, it runs `vercel rollback`
to immediately restore the previous production deployment.

## GitHub secrets

Store these in both the `staging` and `production` GitHub Environments:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

Protect the `production` environment with required reviewers. Require the
`CI/CD / Lint, test, coverage, and Django readiness` check before merging the
production branch.

Vercel itself holds all application secrets. See [VERCEL.md](VERCEL.md) for
the required Vercel environment variables and database migration procedure.
