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
        {
          devShells.default = pkgs.mkShell {
            packages = with pkgs; [ uv ];
            UV_PYTHON = lib.getExe pkgs.python312;
            shellHook = ''
              uv sync --all-extras --locked

              # if .env exists, export its variables
              if [ -f .env ]; then
                set -a
                source .env
                set +a
              fi
            '';
          };
        };
    };
}
