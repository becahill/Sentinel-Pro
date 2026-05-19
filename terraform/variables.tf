variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project name used in resource names."
  type        = string
  default     = "sentinel-pro"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "prod"
}

variable "vpc_cidr" {
  description = "CIDR block for the Sentinel-Pro VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Two or more public subnet CIDRs for ECS, RDS subnet groups, and Redis subnet groups."
  type        = list(string)
  default     = ["10.42.1.0/24", "10.42.2.0/24"]
}

variable "availability_zones" {
  description = "Optional availability zones. Defaults to the first zones available in the selected region."
  type        = list(string)
  default     = []
}

variable "app_image" {
  description = "Container image for the Python API, worker, and dashboard."
  type        = string
}

variable "api_desired_count" {
  description = "Number of API tasks."
  type        = number
  default     = 2
}

variable "worker_desired_count" {
  description = "Number of Celery worker tasks."
  type        = number
  default     = 1
}

variable "api_cpu" {
  description = "Fargate CPU units for the API task."
  type        = number
  default     = 512
}

variable "api_memory" {
  description = "Fargate memory in MiB for the API task."
  type        = number
  default     = 1024
}

variable "worker_cpu" {
  description = "Fargate CPU units for the worker task."
  type        = number
  default     = 512
}

variable "worker_memory" {
  description = "Fargate memory in MiB for the worker task."
  type        = number
  default     = 1024
}

variable "db_name" {
  description = "Postgres database name."
  type        = string
  default     = "sentinel"
}

variable "db_username" {
  description = "Postgres database username."
  type        = string
  default     = "sentinel"
}

variable "db_password" {
  description = "Postgres database password."
  type        = string
  sensitive   = true
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GiB."
  type        = number
  default     = 20
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type."
  type        = string
  default     = "cache.t4g.micro"
}

variable "oauth_clients" {
  description = "SENTINEL_OAUTH_CLIENTS value, preferably JSON or client_id:secret:role entries."
  type        = string
  sensitive   = true
}

variable "jwt_secret" {
  description = "SENTINEL_JWT_SECRET value. Use at least 32 random characters."
  type        = string
  sensitive   = true
}

variable "jwt_access_token_ttl_seconds" {
  description = "OAuth2 access token TTL in seconds."
  type        = number
  default     = 3600
}

variable "allowed_origins" {
  description = "Comma-separated CORS allow-list."
  type        = string
  default     = "http://localhost"
}

variable "log_level" {
  description = "API log level."
  type        = string
  default     = "INFO"
}

variable "tags" {
  description = "Additional tags to apply to AWS resources."
  type        = map(string)
  default     = {}
}
