# Redução dimensional

## Arquiteturas

1. Simples $3^{x}$
   * 9@5x5_s2 - 18@5x5_s2 - 27@5x5_s5 - 100D - 1350D - 27@5x5_s5 - 18@5x5_s2 - 9@5x5_s2
   *  100x50      50x25       10x5     100x1   1350x1   50x25        100x50    200x100 
   * Parameters: 322,634
   * Batch: 64
   * @15 epochs mse: [0.0921, 0.1006, 0.0951]
  
2. Simples $3^{x}$
   * 3@5x5_s2 - 9@5x5_s2 - 27@5x5_s5 - 100D - 1350D - 27@5x5_s5 - 9@5x5_s2 - 3@5x5_s2
   *  100x50      50x25       10x5     100x1   1350x1   50x25        100x50    200x100 
   * Parameters: 303,404
   * Batch: 64
   * @15 epochs mse: [0.1005, 0.1044, 0.1044]
   * @3 epochs mse: [0.3901, 0.4124, 0.3780]
   * 
3. Simples $3^{x}$
   * 3@5x5_s2 - 9@5x5_s2 - 27@5x5_s5 - 2@10x10 - 27@5x5_s5 - 9@5x5_s2 - 3@5x5_s2
   *  100x50      50x25       10x5      10x5      50x25       100x50    200x100 
   * Parameters: 20,481
   * Batch: 64
   * @15 epochs mse: [0.1190, ]
   * @3 epochs mse: [0.335, ]
  
4. Depthwise $3^{x}$ wich layer has a depth wise convolutional layer
   * 3@5x5e3_s2 - 9@5x5e3_s2 - 27@5x5e3_s5 - 27@5x5c3_s5 - 9@5x5c3_s2 - 3@5x5c3_s2
   *  100x50      50x25       10x5     100x1   1350x1   50x25        100x50    200x100 
   * Parameters: 297,554
   * Batch: 64
   * @15 epochs mse = [0.1013, 0.0980, 0.1068]
  
5. Expansion Contraction autoencoder $3^{x}$
   * 3@5x5_s2 - 18@5x5_s2 - 27@5x5_s5 - 100D - 1350D - 27@5x5_s5 - 18@5x5_s2 - 9@5x5_s2
   *  100x50      50x25       10x5     100x1   1350x1   50x25        100x50    200x100 
   * Parameters: 
   * Batch: 64
   * @15 epochs mse = [0.1668, 0.1657, 0.1603]
   * @3 epochs: [0.2369, 0.1992, 0.2313]
  
6. Expansion Contraction autoencoder $3^{x}$
   * 3@5x5_s2 - 18@5x5_s2 - 27@5x5_s5 - 4@5x5  - 27@5x5_s5 - 18@5x5_s2 - 9@5x5_s2
   *  100x50      50x25       10x5       10x5     50x25        100x50    200x100 
   * Parameters: 
   * Batch: 32
   * @15 epochs mse = [0.0685, ]
   * @3 epochs: [0.1195, ]
  
7. Expansion Contraction autoencoder $3^{x}$ wo flat latent
   * 3@5x5e3_s2 - 9@5x5e3_s2 - 27@5x5e3_s5 - 27@5x5c3_s5 - 9@5x5c3_s2 - 3@5x5c3_s2
   *  100x50         50x25         10x5         50x25        100x50       200x100 
   * Parameters: 276,426
   * Batch: 32
   * @60 epochs mse = [0.01135, ]
   * @15 epochs mse = [0.0209, 0,0198, 0.0227]
   * @3 epochs: [0.0498, 0.0424, 0.053]
  
8. Added layer $3^{x}$ 
   * 3@5x5e3_s2 - 9@5x5e3_s2 - 27@5x5e3_s5 - 27@5x5c3_s5 - 9@5x5c3_s2 - 3@5x5c3_s2
   *  100x50         50x25         10x5          50x25        100x50        200x100 
   * Parameters: 1,308,526
   * Batch: 32
   * @15 epochs mse = [0.0736, 0.0638, 0.0756]
   * @3 epochs: [0.3395, 0.2409, 0.2540]
   
9. Added layer $3^{x}$ 
   * 3@5x5_s2 - 9@5x5_s2 - 27@5x5_s5 -  100D -  50D  - 27@1x1_s1 - 27@5x5_s5 - 9@5x5_s2 + 3@5x5_s2
   *  100x50      50x25       10x5     100x1    50x1     10x5       50x25       100x50  + 200x100 
   * 1@1x1_1                         -  100D                        5000D      9@1x1_s1 +
     100x50                            100x1                       5000x1       100x50  +
   * Parameters: 1,177,280
   * Batch: 32
   * @15 epochs mse = [0.0553, 0.0599, 0.0638]
   * @3 epochs: [0.2452, 0.2719, 0.2843]
10. Others added layers
   * @60e-20x20x20_o: 0.0287
   * @60e-40x40x40: 0.0246
   * @60e-40x40x40_o: 0.0313
   * @60e-100x50x10: 0.0264
   * @60e-200x20x2:  0.03
   * @60e-3x9x27:  [0.0364, ]
   * @60e-3x9x27-200D0x100D1:  0.303
   * @60e-3x9x27-10D0x100D1:  0.399
   * @60e-3x9x27-20D0x20D1:  0.1061
   * @60e-3x9x27-50D0x150D1:  0.0236
   * @60e-3x9x27-502D0x50D1 - 1350D:  0.04

