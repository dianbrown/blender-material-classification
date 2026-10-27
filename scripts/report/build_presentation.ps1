param([string]$DeckPath, [switch]$RenderOnly)
$ErrorActionPreference = 'Stop'
$DeckPath = (Resolve-Path -LiteralPath $DeckPath).Path
$deck = Get-Content -LiteralPath $DeckPath -Raw -Encoding UTF8 | ConvertFrom-Json
$outputDir = Split-Path -Parent $DeckPath
$renderDir = Join-Path $outputDir 'renders'
New-Item -ItemType Directory -Path $renderDir -Force | Out-Null
function Color([string]$hex) { return [int]::Parse($hex.Substring(0,2),'HexNumber') + 256*[int]::Parse($hex.Substring(2,2),'HexNumber') + 65536*[int]::Parse($hex.Substring(4,2),'HexNumber') }
function TextBox($slide, $text, $x,$y,$w,$h,$size,$color,$bold) {
    $shape=$slide.Shapes.AddTextbox(1,$x,$y,$w,$h)
    $shape.TextFrame.MarginLeft=0; $shape.TextFrame.MarginRight=0
    $shape.TextFrame.MarginTop=0; $shape.TextFrame.MarginBottom=0
    $shape.TextFrame.WordWrap=-1; $shape.TextFrame.AutoSize=0
    $tr=$shape.TextFrame.TextRange; $tr.Text=$text
    $tr.Font.Name=$deck.font; $tr.Font.Size=$size; $tr.Font.Color.RGB=(Color $color)
    $tr.Font.Bold= $(if($bold){-1}else{0})
    $tr.ParagraphFormat.SpaceAfter=6
    $shape.Left=[single]$x; $shape.Top=[single]$y
    $shape.Width=[single]$w; $shape.Height=[single]$h
    return $shape
}
$app = New-Object -ComObject PowerPoint.Application
if($RenderOnly){
    $presentation=$app.Presentations.Open((Join-Path $outputDir 'material_classification.pptx'),-1,0,0)
    try {
        $presentation.SaveAs((Join-Path $outputDir 'material_classification.pdf'),32)
        foreach($slide in $presentation.Slides){$slide.Export((Join-Path $renderDir ('slide-{0:D2}.png' -f $slide.SlideIndex)),'PNG',1600,900)}
        Write-Output ('Exported PDF and '+$presentation.Slides.Count+' slide renders from final PPTX')
    } finally {$presentation.Close();$app.Quit()}
    exit
}
$presentation = $app.Presentations.Add(0)
$presentation.PageSetup.SlideWidth=$deck.slide_width
$presentation.PageSetup.SlideHeight=$deck.slide_height
$overflows=@()
try {
    $index=0
    foreach($s in $deck.slides) {
        $index++
        $slide=$presentation.Slides.Add($index,12)
        $slide.FollowMasterBackground=0
        $slide.Background.Fill.ForeColor.RGB=(Color 'FAFBFC')
        if($index -eq 1) {
            $null=TextBox $slide $s.title 48 135 860 125 44 '142C40' $true
        } else {
            $null=TextBox $slide $s.title 48 35 870 62 32 '142C40' $true
        }
        $null=TextBox $slide ([string]$index) 900 520 24 15 10 '607383' $false
        foreach($item in $s.items) {
            if($item.kind -eq 'text') {
                $shape=TextBox $slide $item.text $item.x $item.y $item.w $item.h $item.size $item.color $item.bold
                if($item.name){$shape.Name=$item.name}
            } elseif($item.kind -eq 'image') {
                $null=$slide.Shapes.AddPicture((Join-Path $outputDir $item.path),0,-1,$item.x,$item.y,$item.w,$item.h)
            } elseif($item.kind -eq 'table') {
                $table=$slide.Shapes.AddTable($item.data.Count,$item.data[0].Count,$item.x,$item.y,$item.w,$item.h).Table
                if($item.widths){for($c=1;$c -le $table.Columns.Count;$c++){$table.Columns.Item($c).Width=$item.w*$item.widths[$c-1]}}
                for($r=1;$r -le $table.Rows.Count;$r++) {
                    $table.Rows.Item($r).Height=[single]($item.h/$table.Rows.Count)
                    for($c=1;$c -le $table.Columns.Count;$c++) {
                        $shape=$table.Cell($r,$c).Shape
                        $shape.TextFrame.MarginLeft=8; $shape.TextFrame.MarginRight=5
                        $shape.TextFrame.MarginTop=4; $shape.TextFrame.MarginBottom=3
                        $shape.TextFrame.VerticalAnchor=3
                        $tr=$shape.TextFrame.TextRange
                        $tr.Text=[string]$item.data[$r-1][$c-1]
                        $tr.Font.Name=$deck.font; $tr.Font.Size=$item.size
                        $tr.Font.Color.RGB=(Color '20364A'); $tr.Font.Bold=0
                        if($r -eq 1){$shape.Fill.ForeColor.RGB=(Color '20364A');$tr.Font.Color.RGB=(Color 'FFFFFF');$tr.Font.Bold=-1}
                        elseif($item.heat -and $c -gt 1){
                            $v=[double]$item.data[$r-1][$c-1]/90
                            $red=[int](245-225*$v);$green=[int](249-122*$v);$blue=[int](250-120*$v)
                            $shape.Fill.ForeColor.RGB=$red+256*$green+65536*$blue
                            if($v -gt .5){$tr.Font.Color.RGB=(Color 'FFFFFF')}
                            $tr.ParagraphFormat.Alignment=2
                        } else {$shape.Fill.ForeColor.RGB=$(if($r%2 -eq 0){Color 'EDF3F5'}else{Color 'FFFFFF'})}
                    }
                }
            } elseif($item.kind -eq 'chart') {
                # Native chart XML and its embedded workbook are added by the Python
                # builder after PowerPoint saves the editable base presentation.
                continue

            }
        }
        $notes=$s.notes+"`n`nSources:`n"+($s.sources -join "`n")
        $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text=$notes
        foreach($shape in $slide.Shapes){
            if($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1){
                if($shape.TextFrame.TextRange.BoundHeight -gt $shape.Height+2){$overflows+=@{slide=$index;name=$shape.Name;text=$shape.TextFrame.TextRange.Text;boundHeight=$shape.TextFrame.TextRange.BoundHeight;height=$shape.Height}}
            }
        }
        Write-Output "Built slide $index"
    }
    $presentation.SaveAs((Join-Path $outputDir 'qa/base.pptx'),24)
    ConvertTo-Json -InputObject @($overflows) -Depth 5 | Set-Content -LiteralPath (Join-Path $outputDir 'qa/text_overflow.json') -Encoding UTF8
    Write-Output ('Saved editable base with '+$presentation.Slides.Count+' slides')
} finally {
    $presentation.Close()
    $app.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($app)
}
