# Sentinel-Pro Terraform

This module provisions an AWS baseline for Sentinel-Pro:

- VPC, public subnets, route table, and security groups
- RDS Postgres
- ElastiCache Redis
- ECS Fargate API and Celery worker services
- Application Load Balancer for the API
- Secrets Manager values for database, Redis, OAuth2 clients, and JWT signing

## Usage

Create a `terraform.tfvars` file:

```hcl
aws_region = "us-east-1"
app_image  = "123456789012.dkr.ecr.us-east-1.amazonaws.com/sentinel-pro:latest"

db_password = "replace-with-a-random-db-password"
jwt_secret  = "replace-with-at-least-32-random-characters"

oauth_clients = jsonencode({
  "admin-cli" = {
    secret = "replace-with-admin-secret"
    role   = "admin"
  }
  "analyst-ui" = {
    secret = "replace-with-analyst-secret"
    role   = "analyst"
  }
  "ingest-pipeline" = {
    secret = "replace-with-ingest-secret"
    role   = "ingest"
  }
})
```

Then run:

```bash
terraform init
terraform plan
terraform apply
```

The `api_url` output points to the public API load balancer. Exchange OAuth2 client
credentials at `/oauth/token` and send the returned JWT as `Authorization: Bearer <token>`.

This baseline keeps ECS tasks in public subnets so they can pull images without a NAT
gateway. RDS and Redis are not publicly accessible and only accept traffic from the ECS
task security group.
