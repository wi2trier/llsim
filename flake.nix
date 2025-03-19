{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    systems.url = "github:nix-systems/default";
  };
  outputs =
    inputs@{
      flake-parts,
      systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = import systems;
      perSystem =
        {
          pkgs,
          lib,
          ...
        }:
        let
          allDomains = [
            "cars"
            "recipes"
            "arguments"
          ];
          largeModels = [
            "gpt-4o" # $2.5/M input tokens, https://openrouter.ai/openai/gpt-4o
            "o3-mini" # $1.1/M input tokens, https://openrouter.ai/openai/o3-mini
            "deepseek-v3" # $1.2/M input tokens, https://openrouter.ai/deepseek/deepseek-chat
            "deepseek-r1" # $0.8/M input tokens, https://openrouter.ai/deepseek/deepseek-r1
            "llama-405b" # $0.8/M input tokens, https://openrouter.ai/meta-llama/llama-3.1-405b-instruct
            # the models below are not used for evaluation
            # "command-r-plus" # $2.375/M input tokens, https://openrouter.ai/cohere/command-r-plus-08-2024
            # "nova-pro" # $0.8/M input tokens, https://openrouter.ai/amazon/nova-pro-v1
            # "qwen-max" # $1.6/M input tokens, https://openrouter.ai/qwen/qwen-max
            # "claude" # $3/M input tokens, https://openrouter.ai/anthropic/claude-3.7-sonnet
            # "claude-thinking" # $3/M input tokens, https://openrouter.ai/anthropic/claude-3.7-sonnet:thinking
            # "o1" # $15/M input tokens, https://openrouter.ai/openai/o1
            # "gpt-4-5" # $75/M input tokens, https://openrouter.ai/openai/gpt-4.5-preview
          ];
          mediumModels = [
            "gpt-4o-mini" # $0.15/M input tokens, https://openrouter.ai/openai/gpt-4o-mini
            "llama-70b" # $0.12/M input tokens, https://openrouter.ai/meta-llama/llama-3.3-70b-instruct
            # the models below are not used for evaluation
            # "gemini-flash" # $0.1/M input tokens, https://openrouter.ai/google/gemini-2.0-flash-001
            # "command-r" # $0.1425/M input tokens, https://openrouter.ai/cohere/command-r-08-2024
            # "qwen-72b" # $0.13/M input tokens, https://openrouter.ai/qwen/qwen-2.5-72b-instruct
            # "qwen-plus" # $0.4/M input tokens, https://openrouter.ai/qwen/qwen-plus
            # "gemini-flash-lite" # $0.075/M input tokens, https://openrouter.ai/google/gemini-2.0-flash-lite-001
            # "qwen-turbo" # $0.05/M input tokens, https://openrouter.ai/qwen/qwen-turbo
            # "nova-lite" # $0.06/M input tokens, https://openrouter.ai/amazon/nova-lite-v1
          ];
          smallModels = [
            "llama-3b" # $0.015/M input tokens, https://openrouter.ai/meta-llama/llama-3.2-3b-instruct
            # the models below are not used for evaluation
            # "llama-8b" # $0.02/M input tokens, https://openrouter.ai/meta-llama/llama-3.1-8b-instruct
            # "qwen-7b" # $0.025/M input tokens, https://openrouter.ai/qwen/qwen-2.5-7b-instruct
            # "nova-micro" # $0.035/M input tokens, https://openrouter.ai/amazon/nova-micro-v1
            # "gemma-9b" # $0.03/M input tokens, https://openrouter.ai/google/gemma-2-9b-it
            # "command-r7b" # $0.0375/M input tokens, https://openrouter.ai/cohere/command-r7b-12-2024
          ];
          mkEval =
            {
              name,
              combinations,
              mkCombination,
            }:
            let
              patchedMkCombination = combination: ''
                echo
                ${mkCombination combination}
                echo
              '';

            in
            pkgs.writeShellApplication {
              inherit name;
              runtimeInputs = with pkgs; [ uv ];
              runtimeEnv = {
                UV_PYTHON = lib.getExe pkgs.python312;
                OLLAMA_HOST = "http://gpu.wi2.uni-trier.de:5000";
              };
              text = ''
                set -x # echo on
                uv sync --all-extras --locked

                ${lib.concatLines (map patchedMkCombination combinations)}
              '';
            };
        in
        {
          packages = {
            retrieve-baseline = mkEval {
              name = "retrieve-baseline";
              combinations = lib.cartesianProduct {
                domain = allDomains;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  --domain ${attrs.domain} \
                  --out "data/output/${attrs.domain}/baseline.json"
              '';
            };
            retrieve-naive = mkEval {
              name = "retrieve-naive";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                variant = [
                  "Sim"
                  "Rank"
                ];
                model = mediumModels;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  --domain ${attrs.domain} \
                  --retriever llsim.naive:${attrs.variant}Retriever \
                  --retriever-arg model=${attrs.model} \
                  --out "data/output/${attrs.domain}/naive-${lib.toLower attrs.variant}-${attrs.model}.json"
              '';
            };
            build-preferences-medium = mkEval {
              name = "build-preferences-medium";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = mediumModels;
              };
              mkCombination = attrs: ''
                uv run llsim build-preferences "$@" \
                  --domain ${attrs.domain} \
                  --model ${attrs.model} \
                  --out "data/output/${attrs.domain}/preferences-${attrs.model}-config.json"
              '';
            };
            build-preferences-small = mkEval {
              name = "build-preferences-small";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = smallModels;
              };
              mkCombination = attrs: ''
                uv run llsim build-preferences "$@" \
                  --domain ${attrs.domain} \
                  --model ${attrs.model} \
                  --pairwise \
                  --out "data/output/${attrs.domain}/preferences-${attrs.model}-config.json"
              '';
            };
            retrieve-preferences = mkEval {
              name = "retrieve-preferences";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = mediumModels ++ smallModels;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  --domain ${attrs.domain} \
                  --retriever llsim.centrality:Retriever \
                  --retriever-arg file="data/output/${attrs.domain}/preferences-${attrs.model}-config.json" \
                  --retriever-arg measures=pagerank,hits \
                  --out "data/output/${attrs.domain}/preferences-${attrs.model}.json"
              '';
            };
            build-similarity = mkEval {
              name = "build-similarity";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = largeModels;
              };
              mkCombination = attrs: ''
                uv run llsim build-similarity "$@" \
                  --domain ${attrs.domain} \
                  --model ${attrs.model} \
                  ${
                    lib.optionalString (
                      attrs.domain == "arguments" || attrs.domain == "recipes"
                    ) "--attribute-table type"
                  } \
                  --out "data/output/${attrs.domain}/builder-${attrs.model}-config.json"
              '';
            };
            retrieve-builder = mkEval {
              name = "retrieve-builder";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = largeModels;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  --domain ${attrs.domain} \
                  --retriever llsim.builder:Retriever \
                  --retriever-arg file="data/output/${attrs.domain}/builder-${attrs.model}-config.json" \
                  --out "data/output/${attrs.domain}/builder-${attrs.model}.json"
              '';
            };
            evaluate-run = mkEval {
              name = "evaluate-run";
              combinations = lib.cartesianProduct {
                domain = allDomains;
              };
              mkCombination = attrs: ''
                uv run llsim evaluate-run "$@" \
                  "data/output/${attrs.domain}"
              '';
            };
            evaluate-qrels = mkEval {
              name = "evaluate-qrels";
              combinations = lib.cartesianProduct {
                domain = [ "arguments" ];
              };
              mkCombination = attrs: ''
                uv run llsim evaluate-qrels "$@" \
                  --domain ${attrs.domain} \
                  "data/output/${attrs.domain}"
              '';
            };
          };
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [ uv ];
            UV_PYTHON = lib.getExe pkgs.python312;
            TOKENIZERS_PARALLELISM = true;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
