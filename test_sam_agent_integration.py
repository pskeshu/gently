#!/usr/bin/env python3
"""
Test SAM Detection Integration with Agent

Simple test of the SAM + Claude embryo detection through the conversational agent.
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware, run_rich_cli


async def test_detection_simple():
    """Test detection and get embryo positions"""
    print("="*70)
    print("SAM DETECTION INTEGRATION TEST")
    print("="*70)

    # Create copilot with hardware
    copilot = create_copilot_with_hardware(
        storage_path=Path("./test_sam_detection")
    )

    # Just ask the agent to detect embryos
    print("\nAsking agent to detect embryos...")
    response = await copilot.handle_message("Find all embryos automatically")

    print("\n" + "="*70)
    print("AGENT RESPONSE:")
    print("="*70)
    print(response)

    # Check what embryos were detected
    print("\n" + "="*70)
    print("LOADED EMBRYOS:")
    print("="*70)

    for embryo_id, embryo in copilot.experiment.embryos.items():
        print(f"\n{embryo_id}:")
        print(f"  Position: ({embryo.position['x']:.1f}, {embryo.position['y']:.1f}) µm")

    return copilot.experiment.embryos


async def test_interactive():
    """Interactive test with Rich CLI"""
    print("="*70)
    print("INTERACTIVE SAM DETECTION TEST")
    print("="*70)

    copilot = create_copilot_with_hardware(
        storage_path=Path("./test_sam_detection")
    )

    print("\n✓ Copilot ready with hardware control")
    print("\nTry these commands:")
    print('  • "Find all embryos"')
    print('  • "Show me embryo positions"')
    print('  • "Calibrate embryo_000"')
    print('  • "/embryos" to see loaded embryos')
    print()

    await run_rich_cli(copilot, history_file=Path("./test_sam_detection/.history"))


async def test_standalone_detector():
    """Test SAM detector without agent (direct use)"""
    from gently.agent import SAMEmbryoDetector
    from client import get_mmc
    import numpy as np

    print("="*70)
    print("STANDALONE SAM DETECTOR TEST")
    print("="*70)

    # Initialize detector
    detector = SAMEmbryoDetector(
        sam_checkpoint="sam_vit_b_01ec64.pth",
        sam_model_type="vit_b"
    )

    # Capture image from bottom camera
    core = get_mmc()
    core.setCameraDevice("Bottom PCO")
    core.setExposure("Bottom PCO", 50.0)
    core.snapImage()
    image = core.getImage()

    # Get stage position
    xy_stage = core.getXYStageDevice()
    stage_x = core.getXPosition(xy_stage)
    stage_y = core.getYPosition(xy_stage)
    stage_pos = (stage_x, stage_y)

    print(f"\nStage position: ({stage_x:.1f}, {stage_y:.1f}) µm")
    print(f"Image shape: {image.shape}")

    # Run detection
    results = await detector.detect_embryos(
        image=image,
        stage_position=stage_pos,
        use_claude_review=True,
        save_visualizations=True
    )

    # Show results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)

    print(f"\nInitial detections: {results['initial_detections']}")
    print(f"Final detections: {results['final_detections']}")
    print(f"Claude verified: {results['verification'].get('verified', False)}")

    print("\nEmbryo positions:")
    for embryo in results['embryos']:
        print(f"  {embryo['embryo_id']}: ({embryo['stage_x_um']:.1f}, {embryo['stage_y_um']:.1f}) µm")
        print(f"     Pixel: ({embryo['pixel_x']:.0f}, {embryo['pixel_y']:.0f})")
        print(f"     Confidence: {embryo['confidence']:.2f}")

    print(f"\nImages saved to: {results['images']['final']}")

    # Show in napari
    try:
        detector.show_in_napari(image, results['embryos'], block=True)
    except Exception as e:
        print(f"\nCould not show in napari: {e}")

    return results


if __name__ == "__main__":
    import sys
    import os

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key'")
        exit(1)

    # Choose test mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "interactive"

    if mode == "simple":
        # Simple detection test
        asyncio.run(test_detection_simple())
    elif mode == "standalone":
        # Test SAM detector directly
        asyncio.run(test_standalone_detector())
    else:
        # Interactive CLI
        asyncio.run(test_interactive())
