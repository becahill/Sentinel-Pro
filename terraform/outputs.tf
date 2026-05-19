output "api_url" {
  description = "Public Sentinel-Pro API URL."
  value       = "http://${aws_lb.api.dns_name}"
}

output "ecs_cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.this.name
}

output "postgres_endpoint" {
  description = "RDS Postgres endpoint."
  value       = aws_db_instance.postgres.address
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint."
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
}

output "secrets_manager_secret_arn" {
  description = "Secrets Manager secret containing app runtime configuration."
  value       = aws_secretsmanager_secret.app.arn
}
