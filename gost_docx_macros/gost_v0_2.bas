Sub GOST()
    Dim doc As Document
    Set doc = ActiveDocument
    
    With doc.PageSetup
        .Orientation = wdOrientPortrait
        .TopMargin = CentimetersToPoints(2)
        .BottomMargin = CentimetersToPoints(2)
        .LeftMargin = CentimetersToPoints(3)
        .RightMargin = CentimetersToPoints(1)
        .Gutter = 0
        .HeaderDistance = CentimetersToPoints(1.25)
        .FooterDistance = CentimetersToPoints(1.25)
    End With
    
    doc.Content.Font.Name = "Times New Roman"
    doc.Content.Font.Size = 14
    
    With doc.Styles(wdStyleNormal).Font
        .Name = "Times New Roman"
        .Size = 14
    End With
    
    With doc.Styles(wdStyleHeading1).Font
        .Name = "Times New Roman"
        .Size = 14
        .Bold = True
    End With
    doc.Styles(wdStyleHeading1).ParagraphFormat.Alignment = wdAlignParagraphCenter
    doc.Styles(wdStyleHeading1).ParagraphFormat.SpaceAfter = 12
    
    With doc.Styles(wdStyleHeading2).Font
        .Name = "Times New Roman"
        .Size = 14
        .Bold = True
    End With
    doc.Styles(wdStyleHeading2).ParagraphFormat.Alignment = wdAlignParagraphLeft
    doc.Styles(wdStyleHeading2).ParagraphFormat.SpaceAfter = 12
    
    With doc.Styles(wdStyleHeading3).Font
        .Name = "Times New Roman"
        .Size = 14
        .Bold = True
    End With
    doc.Styles(wdStyleHeading3).ParagraphFormat.Alignment = wdAlignParagraphLeft
    doc.Styles(wdStyleHeading3).ParagraphFormat.SpaceAfter = 12
    
    ' Force update of all Heading 3 text
    Dim para As Paragraph
    For Each para In doc.Paragraphs
        If para.Style = doc.Styles(wdStyleHeading3) Then
            para.Range.Font.Name = "Times New Roman"
            para.Range.Font.Size = 14
            para.Range.Font.Bold = True
            para.Format.Alignment = wdAlignParagraphLeft
            para.Format.SpaceAfter = 12
        End If
    Next para
    
    With doc.Styles(wdStyleNormal).ParagraphFormat
        .Alignment = wdAlignParagraphJustify
        .FirstLineIndent = CentimetersToPoints(1.25)
        .LineSpacingRule = wdLineSpace1pt5
        .SpaceAfter = 0
    End With
    
    Dim footer As HeaderFooter
    Set footer = doc.Sections(1).Footers(wdHeaderFooterPrimary)
    
    With footer
        .PageNumbers.Add PageNumberAlignment:=wdAlignPageNumberRight
        .LinkToPrevious = False
    End With
    
    On Error Resume Next
    With doc.Styles("Bibliography").ParagraphFormat
        .LeftIndent = CentimetersToPoints(1.25)
        .FirstLineIndent = CentimetersToPoints(-0.5)
        .Alignment = wdAlignParagraphLeft
    End With

    doc.Styles("Bibliography").ParagraphFormat.Alignment = wdAlignParagraphLeft
    doc.Styles("Bibliography").ParagraphFormat.FirstLineIndent = 0
    doc.Styles("Bibliography").ParagraphFormat.LeftIndent = CentimetersToPoints(1.25)
    On Error GoTo 0
    
    MsgBox "formatted", vbInformation
End Sub

