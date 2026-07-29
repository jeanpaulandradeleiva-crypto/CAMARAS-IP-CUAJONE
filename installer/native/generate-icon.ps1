# SPDX-License-Identifier: AGPL-3.0-only

[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$parent = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $parent -PathType Container)) {
    throw "Icon output parent does not exist: $parent"
}

Add-Type -AssemblyName System.Drawing
$sizes = @(16, 32, 48, 256)
$pngPayloads = [System.Collections.Generic.List[byte[]]]::new()

foreach ($size in $sizes) {
    $bitmap = [System.Drawing.Bitmap]::new($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.Clear([System.Drawing.Color]::Transparent)
        $navy = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 17, 35, 58))
        $copper = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(255, 224, 125, 54))
        $whitePen = [System.Drawing.Pen]::new([System.Drawing.Color]::White, [Math]::Max(1.5, $size * 0.075))
        $whitePen.StartCap = [System.Drawing.Drawing2D.LineCap]::Round
        $whitePen.EndCap = [System.Drawing.Drawing2D.LineCap]::Round
        $whitePen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
        try {
            $graphics.FillEllipse($navy, 0, 0, $size - 1, $size - 1)
            $points = [System.Drawing.PointF[]]@(
                [System.Drawing.PointF]::new($size * 0.50, $size * 0.16),
                [System.Drawing.PointF]::new($size * 0.79, $size * 0.27),
                [System.Drawing.PointF]::new($size * 0.74, $size * 0.64),
                [System.Drawing.PointF]::new($size * 0.50, $size * 0.84),
                [System.Drawing.PointF]::new($size * 0.26, $size * 0.64),
                [System.Drawing.PointF]::new($size * 0.21, $size * 0.27)
            )
            $graphics.FillPolygon($copper, $points)
            $check = [System.Drawing.PointF[]]@(
                [System.Drawing.PointF]::new($size * 0.34, $size * 0.50),
                [System.Drawing.PointF]::new($size * 0.46, $size * 0.63),
                [System.Drawing.PointF]::new($size * 0.68, $size * 0.38)
            )
            $graphics.DrawLines($whitePen, $check)
        }
        finally {
            $whitePen.Dispose()
            $copper.Dispose()
            $navy.Dispose()
        }
    }
    finally {
        $graphics.Dispose()
    }

    $stream = [System.IO.MemoryStream]::new()
    try {
        $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
        $pngPayloads.Add($stream.ToArray())
    }
    finally {
        $stream.Dispose()
        $bitmap.Dispose()
    }
}

$file = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::Create, [System.IO.FileAccess]::Write)
$writer = [System.IO.BinaryWriter]::new($file)
try {
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]$sizes.Count)
    $offset = 6 + (16 * $sizes.Count)
    for ($index = 0; $index -lt $sizes.Count; $index++) {
        $size = $sizes[$index]
        $writer.Write([byte]$(if ($size -eq 256) { 0 } else { $size }))
        $writer.Write([byte]$(if ($size -eq 256) { 0 } else { $size }))
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]32)
        $writer.Write([uint32]$pngPayloads[$index].Length)
        $writer.Write([uint32]$offset)
        $offset += $pngPayloads[$index].Length
    }
    foreach ($payload in $pngPayloads) {
        $writer.Write($payload)
    }
}
finally {
    $writer.Dispose()
    $file.Dispose()
}
