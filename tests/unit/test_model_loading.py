"""Unit tests for model loading functionality."""


def test_pretrained_models_exist(project_root):
    """Test that pretrained model directories exist."""
    pretrained_dir = project_root / "data" / "pretrained"
    assert pretrained_dir.exists(), "Pretrained models directory should exist"

    # Check for AOT-GAN models
    aot_gan_dir = pretrained_dir / "aot-gan"
    assert aot_gan_dir.exists(), "AOT-GAN models directory should exist"

    # Check for ICT models
    ict_dir = pretrained_dir / "ict"
    assert ict_dir.exists(), "ICT models directory should exist"

    # Check for RePaint models
    repaint_dir = pretrained_dir / "repaint"
    assert repaint_dir.exists(), "RePaint models directory should exist"


def test_aot_gan_model_variants(project_root):
    """Test that AOT-GAN model variants exist."""
    aot_gan_dir = project_root / "data" / "pretrained" / "aot-gan"

    # Check CelebA-HQ variant
    celebahq_dir = aot_gan_dir / "celebahq"
    assert celebahq_dir.exists(), "AOT-GAN CelebA-HQ models should exist"

    # Check Places2 variant
    places2_dir = aot_gan_dir / "places2"
    assert places2_dir.exists(), "AOT-GAN Places2 models should exist"

    # Check that each variant has required model files
    for variant_dir in [celebahq_dir, places2_dir]:
        generator_file = variant_dir / "G0000000.pt"
        discriminator_file = variant_dir / "D0000000.pt"
        optimizer_file = variant_dir / "O0000000.pt"

        assert generator_file.exists(), f"Generator model missing in {variant_dir.name}"
        assert discriminator_file.exists(), (
            f"Discriminator model missing in {variant_dir.name}"
        )
        assert optimizer_file.exists(), f"Optimizer model missing in {variant_dir.name}"


def test_ict_model_variants(project_root):
    """Test that ICT model variants exist."""
    ict_dir = project_root / "data" / "pretrained" / "ict"

    # Check Upsample variants
    upsample_dir = ict_dir / "Upsample"
    assert upsample_dir.exists(), "ICT Upsample models should exist"

    # Check for FFHQ variant
    ffhq_dir = upsample_dir / "FFHQ"
    assert ffhq_dir.exists(), "ICT FFHQ models should exist"

    # Check for ImageNet variant
    imagenet_dir = upsample_dir / "ImageNet"
    assert imagenet_dir.exists(), "ICT ImageNet models should exist"

    # Check for Places2_Nature variant
    places2_nature_dir = upsample_dir / "Places2_Nature"
    assert places2_nature_dir.exists(), "ICT Places2_Nature models should exist"

    # Check that each variant has required model files
    for variant_dir in [ffhq_dir, imagenet_dir, places2_nature_dir]:
        generator_file = variant_dir / "InpaintingModel_gen.pth"
        discriminator_file = variant_dir / "InpaintingModel_dis.pth"

        assert generator_file.exists(), f"Generator model missing in {variant_dir.name}"
        assert discriminator_file.exists(), (
            f"Discriminator model missing in {variant_dir.name}"
        )


def test_repaint_model_variants(project_root):
    """Test that RePaint model variants exist."""
    repaint_dir = project_root / "data" / "pretrained" / "repaint"

    # Check for CelebA-HQ model
    celebahq_model = repaint_dir / "celeba256_250000.pt"
    assert celebahq_model.exists(), "RePaint CelebA-HQ model should exist"

    # Check for ImageNet model
    imagenet_model = repaint_dir / "256x256_diffusion.pt"
    assert imagenet_model.exists(), "RePaint ImageNet model should exist"

    # Check for Places2 model
    places2_model = repaint_dir / "places256_300000.pt"
    assert places2_model.exists(), "RePaint Places2 model should exist"

    # Check for classifier
    classifier_model = repaint_dir / "256x256_classifier.pt"
    assert classifier_model.exists(), "RePaint classifier model should exist"


def test_output_directories_exist(project_root):
    """Test that output directories exist."""
    output_dir = project_root / "output"
    assert output_dir.exists(), "Output directory should exist"

    # Check for model-specific output directories
    models = ["AOT-GAN", "ICT", "RePaint"]
    for model in models:
        model_output_dir = output_dir / model
        assert model_output_dir.exists(), f"Output directory for {model} should exist"

        # Check for variant subdirectories
        if model == "AOT-GAN":
            variants = ["CelebA-HQ", "Places2", "OAI"]
        elif model == "ICT":
            variants = ["FFHQ", "ImageNet", "Places2_Nature", "OAI"]
        elif model == "RePaint":
            variants = ["CelebA-HQ", "ImageNet", "Places2"]

        for variant in variants:
            variant_dir = model_output_dir / variant
            assert variant_dir.exists(), (
                f"Output directory for {model}/{variant} should exist"
            )

            # Check for subset_4 subdirectory
            subset_4_dir = variant_dir / "subset_4"
            assert subset_4_dir.exists(), (
                f"subset_4 directory for {model}/{variant} should exist"
            )
