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
        in
        {
          packages = {
            eval-plain =
              let
                combinations = lib.cartesianProduct {
                  retriever = [
                    "llsim.plain:SIM_RETRIEVER"
                    "llsim.plain:RANK_RETRIEVER"
                  ];
                  domain = [
                    "cars"
                    "recipes"
                    "arguments"
                  ];
                  model = [
                    "o3-mini"
                    "4o-mini"
                  ];
                };
              in
              pkgs.writeShellApplication {
                name = "eval-plain";
                runtimeInputs = with pkgs; [ uv ];
                runtimeEnv = {
                  UV_PYTHON = lib.getExe pkgs.python312;
                  OLLAMA_HOST = "http://gpu.wi2.uni-trier.de:5000";
                };
                text = ''
                  set -x # echo on
                  uv sync --all-extras --locked

                  ${lib.concatLines (
                    map (attrs: ''
                      uv run llsim retrieve "$@" \
                        ${mkCli attrs} \
                        --out "data/output/${attrs.domain}/${getPythonName attrs.retriever}-${attrs.model}.json"
                    '') combinations
                  )}
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
