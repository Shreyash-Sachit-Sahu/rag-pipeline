# infra/terraform/main.tf
# ────────────────────────────────────────────────────────────
# RAG Pipeline — AWS ECS + Fargate Infrastructure
# All values are parameterised via variables (no hard-coded values).
# Override with terraform.tfvars or environment variables (TF_VAR_*).
# ────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    # Configure via -backend-config or terraform.tfvars
    # bucket = var.tf_state_bucket
    # key    = "rag-pipeline/terraform.tfstate"
    # region = var.aws_region
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ─────────────────────────────────────────────────

variable "aws_region"          { type = string }
variable "project_name"        { type = string; default = "rag-pipeline" }
variable "environment"         { type = string; default = "production" }
variable "ecr_registry"        { type = string; description = "ECR registry URL" }
variable "gateway_image_tag"   { type = string; default = "latest" }
variable "embedder_image_tag"  { type = string; default = "latest" }
variable "retriever_image_tag" { type = string; default = "latest" }
variable "llm_image_tag"       { type = string; default = "latest" }

variable "gateway_cpu"         { type = number; default = 1024 }
variable "gateway_memory"      { type = number; default = 2048 }
variable "llm_cpu"             { type = number; default = 4096 }
variable "llm_memory"          { type = number; default = 16384 }
variable "embedder_cpu"        { type = number; default = 1024 }
variable "embedder_memory"     { type = number; default = 2048 }
variable "retriever_cpu"       { type = number; default = 512 }
variable "retriever_memory"    { type = number; default = 1024 }

variable "redis_node_type"     { type = string; default = "cache.t3.medium" }
variable "alert_p99_ms"        { type = number; default = 150 }
variable "vpc_cidr"            { type = string; default = "10.0.0.0/16" }

# ── Data sources ──────────────────────────────────────────────

data "aws_availability_zones" "available" { state = "available" }
data "aws_caller_identity" "current" {}

# ── VPC ───────────────────────────────────────────────────────

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = { Name = "${var.project_name}-vpc" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Name = "${var.project_name}-private-${count.index}" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${var.project_name}-public-${count.index}" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${var.project_name}-igw" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── ECS Cluster ───────────────────────────────────────────────

resource "aws_ecs_cluster" "main" {
  name = "${var.project_name}-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { Environment = var.environment }
}

# ── ECR Repositories ──────────────────────────────────────────

locals {
  services = ["gateway", "embedder", "retriever", "llm", "ingestion"]
}

resource "aws_ecr_repository" "services" {
  for_each             = toset(local.services)
  name                 = "${var.project_name}/${each.key}"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# ── IAM roles ─────────────────────────────────────────────────

resource "aws_iam_role" "ecs_task_execution" {
  name = "${var.project_name}-ecs-task-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# ── CloudWatch Log Groups ─────────────────────────────────────

resource "aws_cloudwatch_log_group" "services" {
  for_each          = toset(local.services)
  name              = "/ecs/${var.project_name}/${each.key}"
  retention_in_days = 30
}

# ── ElastiCache Redis ─────────────────────────────────────────

resource "aws_elasticache_subnet_group" "redis" {
  name       = "${var.project_name}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_security_group" "redis" {
  name   = "${var.project_name}-redis-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id       = "${var.project_name}-redis"
  description                = "Semantic cache for RAG pipeline"
  node_type                  = var.redis_node_type
  port                       = 6379
  parameter_group_name       = "default.redis7"
  subnet_group_name          = aws_elasticache_subnet_group.redis.name
  security_group_ids         = [aws_security_group.redis.id]
  automatic_failover_enabled = false
  num_cache_clusters         = 1
  tags                       = { Environment = var.environment }
}

# ── ALB ───────────────────────────────────────────────────────

resource "aws_security_group" "alb" {
  name   = "${var.project_name}-alb-sg"
  vpc_id = aws_vpc.main.id
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_lb" "main" {
  name               = "${var.project_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "gateway" {
  name        = "${var.project_name}-gateway"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.gateway.arn
  }
}

# ── ECS Gateway Task Definition ───────────────────────────────

resource "aws_ecs_task_definition" "gateway" {
  family                   = "${var.project_name}-gateway"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.gateway_cpu
  memory                   = var.gateway_memory
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn

  container_definitions = jsonencode([{
    name  = "gateway"
    image = "${var.ecr_registry}/${var.project_name}/gateway:${var.gateway_image_tag}"
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "EMBEDDER_HOST",   value = "embedder.${var.project_name}.local" },
      { name = "RETRIEVER_HOST",  value = "retriever.${var.project_name}.local" },
      { name = "CACHE_HOST",      value = aws_elasticache_replication_group.redis.primary_endpoint_address },
      { name = "LLM_HOST",        value = "llm.${var.project_name}.local" },
      { name = "GATEWAY_PORT",    value = "8000" }
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = aws_cloudwatch_log_group.services["gateway"].name
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "ecs"
      }
    }
    healthCheck = {
      command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
      interval    = 30
      timeout     = 5
      retries     = 3
      startPeriod = 60
    }
  }])
}

resource "aws_ecs_service" "gateway" {
  name            = "${var.project_name}-gateway"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.gateway.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.gateway.arn
    container_name   = "gateway"
    container_port   = 8000
  }
}

# ── CloudWatch Alarms ─────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.project_name}-high-p99-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "TargetResponseTime"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "p99"
  threshold           = var.alert_p99_ms / 1000.0   # convert ms to seconds
  alarm_description   = "P99 latency exceeded ${var.alert_p99_ms}ms"
  alarm_actions       = []  # Add SNS ARN for PagerDuty integration

  dimensions = {
    LoadBalancer = aws_lb.main.arn_suffix
    TargetGroup  = aws_lb_target_group.gateway.arn_suffix
  }
}

# ── Outputs ───────────────────────────────────────────────────

output "alb_dns_name" {
  value       = aws_lb.main.dns_name
  description = "Public DNS name for the ALB (query gateway entry point)"
}

output "redis_endpoint" {
  value       = aws_elasticache_replication_group.redis.primary_endpoint_address
  description = "ElastiCache Redis primary endpoint"
  sensitive   = false
}

output "ecr_repository_urls" {
  value       = { for k, v in aws_ecr_repository.services : k => v.repository_url }
  description = "ECR repository URLs for each service"
}
