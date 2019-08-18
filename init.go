package gofaces

import "github.com/kelseyhightower/envconfig"

func init() {
	err := envconfig.Process(envConfPrefix, &config)
	if err != nil {
		panic(err)
	}
}
