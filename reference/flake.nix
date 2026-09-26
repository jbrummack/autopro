{
  description = "Reference implementations for autopro's tests: HF processors (default, no torch), torchvision + scikit-learn + OpenCV (post; CPU torch, no CUDA)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "aarch64-darwin" "x86_64-darwin" "x86_64-linux" "aarch64-linux" ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShellNoCC {
          packages = [
            (pkgs.python3.withPackages (ps: [
              ps.transformers ps.tokenizers ps.sentencepiece ps.pillow ps.numpy ps.soundfile ps.protobuf
            ]))
          ];
        };
        # Postprocessing references: torchvision NMS, scikit-learn DBSCAN, OpenCV (OCR).
        post = pkgs.mkShellNoCC {
          packages = [
            (pkgs.python3.withPackages (ps: [ ps.torch ps.torchvision ps.scikit-learn ps.numpy ps.opencv4 ]))
          ];
        };
      });
    };
}
