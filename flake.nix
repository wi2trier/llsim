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
          mkCli = lib.cli.toGNUCommandLineShell { };
          getPythonName =
            name:
            lib.pipe name [
              (lib.splitString ":")
              lib.last
              lib.toLower
            ];
          allDomains = [
            "cars"
            "recipes"
            "arguments"
          ];
          reasoningModels = [
            "o1"
            "o3-mini"
            "deepseek-r1"
            "claude-thinking"
          ];
          largeModels = [
            "4o"
            "claude"
            "llama-405b"
            "qwen-max"
            "deepseek-v3"
          ];
          smallModels = [
            "4o-mini"
            "gemini-flash"
            "qwen-plus"
            "llama-70b"
          ];
          tinyModels = [
            "gemini-flash-lite"
            "qwen-turbo"
            "llama-8b"
            "llama-3b"
          ];
          mkEval =
            {
              name,
              combinations,
              mkCombination,
            }:
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

                ${lib.concatLines (map mkCombination combinations)}
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
                  ${mkCli attrs} \
                  --out "data/output/${attrs.domain}/baseline.json"
              '';
            };
            retrieve-plain = mkEval {
              name = "retrieve-plain";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                retriever = [
                  "llsim.plain:SIM_RETRIEVER"
                  "llsim.plain:RANK_RETRIEVER"
                ];
                model = smallModels;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  ${mkCli attrs} \
                  --out "data/output/${attrs.domain}/${getPythonName attrs.retriever}-${attrs.model}.json"
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
                  ${mkCli attrs} \
                  --out "data/output/${attrs.domain}/preferences-${attrs.model}.json"
              '';
            };
            build-preferences-tiny = mkEval {
              name = "build-preferences-tiny";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = tinyModels;
              };
              mkCombination = attrs: ''
                uv run llsim build-preferences "$@" \
                  ${mkCli attrs} \
                  --pairwise \
                  --out "data/output/${attrs.domain}/centrality-${attrs.model}-config.json"
              '';
            };
            retrieve-centrality = mkEval {
              name = "retrieve-centrality";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = smallModels ++ tinyModels;
              };
              mkCombination = attrs: ''
                uv run llsim retrieve "$@" \
                  --domain ${attrs.domain} \
                  --retriever llsim.centrality:Retriever \
                  --retriever-arg file="data/output/${attrs.domain}/centrality-${attrs.model}-config.json" \
                  --retriever-arg measures=pagerank,hits \
                  --out "data/output/${attrs.domain}/centrality-${attrs.model}.json"
              '';
            };
            build-similarity = mkEval {
              name = "build-similarity";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = reasoningModels ++ largeModels;
              };
              mkCombination = attrs: ''
                uv run llsim build-similarity "$@" \
                  ${mkCli attrs} \
                  --out "data/output/${attrs.domain}/builder-${attrs.model}-config.json"
              '';
            };
            retrieve-builder = mkEval {
              name = "retrieve-builder";
              combinations = lib.cartesianProduct {
                domain = allDomains;
                model = reasoningModels ++ largeModels;
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
                domain = allDomains;
              };
              mkCombination = attrs: ''
                uv run llsim evaluate-qrels "$@" \
                  ${mkCli attrs} \
                  "data/output/${attrs.domain}"
              '';
            };
          };
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [ uv ];
            UV_PYTHON = lib.getExe pkgs.python312;
            shellHook = ''
              uv sync --all-extras --locked
            '';
          };
        };
    };
}
