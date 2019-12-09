from pdf2image import convert_from_path
pages = convert_from_path('pernull2 No.1-14.PDF', 500)

for i, page in enumerate(pages):
    page.save('out_7_' + str(i) + '.jpg', "JPEG")

    
