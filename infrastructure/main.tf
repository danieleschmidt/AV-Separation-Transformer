
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
  }
}

# Multi-region deployment
module "av_separation_us_east" {
  source = "./modules/av-separation"
  region = "us-east-1"
  environment = "production"
  instance_type = "p3.2xlarge"
  min_capacity = 2
  max_capacity = 20
}

module "av_separation_eu_west" {
  source = "./modules/av-separation"
  region = "eu-west-1"
  environment = "production"
  instance_type = "p3.2xlarge"
  min_capacity = 2
  max_capacity = 15
}

module "av_separation_ap_southeast" {
  source = "./modules/av-separation"
  region = "ap-southeast-1"
  environment = "production"
  instance_type = "p3.2xlarge"
  min_capacity = 1
  max_capacity = 10
}

# Global load balancer
resource "aws_route53_zone" "main" {
  name = "av-separation.ai"
}

resource "aws_route53_record" "api" {
  zone_id = aws_route53_zone.main.zone_id
  name    = "api.av-separation.ai"
  type    = "A"
  
  set_identifier = "primary"
  
  weighted_routing_policy {
    weight = 100
  }
  
  alias {
    name                   = module.av_separation_us_east.load_balancer_dns
    zone_id                = module.av_separation_us_east.load_balancer_zone_id
    evaluate_target_health = true
  }
}

# Output global endpoints
output "global_endpoints" {
  value = {
    us_east_1    = module.av_separation_us_east.endpoint
    eu_west_1    = module.av_separation_eu_west.endpoint
    ap_southeast_1 = module.av_separation_ap_southeast.endpoint
    global_api   = "https://api.av-separation.ai"
  }
}
