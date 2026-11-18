{
  description = "Game.2048 - Zig core-first 2048 project";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        package = pkgs.stdenvNoCC.mkDerivation {
          pname = "game-2048";
          version = "0.1.0";
          src = ./.;
          strictDeps = true;
          nativeBuildInputs = [ pkgs.zig ];
          dontConfigure = true;
          buildPhase = ''
            runHook preBuild
            export HOME="$TMPDIR"
            zig build -Doptimize=ReleaseSafe -p "$out"
            runHook postBuild
          '';
          installPhase = "true";
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.zig
            pkgs.shellcheck
            pkgs.ruff
            pkgs.shfmt
            pkgs.nodejs_22
            pkgs.python3
          ] ++ pkgs.lib.optionals (pkgs ? zls) [ pkgs.zls ];

          shellHook = ''
            export ZIG_GLOBAL_CACHE_DIR="$PWD/.zig-global-cache"
            if command -v git >/dev/null 2>&1 && [ -d .git ]; then
              if [ "$(git config --get core.hooksPath || true)" != ".githooks" ]; then
                git config core.hooksPath .githooks
                echo "hooks: core.hooksPath=.githooks"
              fi
            fi
            echo "Entered Game.2048 devShell (zig: $(zig version))"
          '';
        };

        packages.default = package;

        apps.default = flake-utils.lib.mkApp {
          drv = package;
          exePath = "/bin/game-2048";
        };

        checks.default = package;
      });
}
