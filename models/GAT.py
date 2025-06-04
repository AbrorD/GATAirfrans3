# models/GAT.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as nng

class GAT(nn.Module):
    def __init__(self, hparams, encoder, decoder):
        super(GAT, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder

        dim_encoded = hparams['encoder'][-1]
        dim_before_decoder = hparams['decoder'][0]

        self.gat_hidden_channels_per_head = hparams.get('gat_hidden_channels_per_head', 64) # Ubah nama agar lebih jelas
        self.gat_layers = hparams.get('gat_layers', 2)
        self.heads = hparams.get('heads', 4)
        self.dropout_gat = hparams.get('dropout_gat', 0.1)
        
        # Kontrol concat untuk layer tersembunyi dan output secara terpisah
        self.concat_hidden_gat = hparams.get('concat_hidden_gat', True) # Default True untuk hidden
        self.concat_output_gat = hparams.get('concat_output_gat', True) # Default True untuk output (seperti sebelumnya)


        self.convs = nn.ModuleList()
        current_dim_input = dim_encoded

        for i in range(self.gat_layers):
            is_last_layer = (i == self.gat_layers - 1)
            
            # Tentukan out_channels dan concat untuk layer saat ini
            if is_last_layer:
                # Untuk layer terakhir, output harus cocok dengan dim_before_decoder
                if self.concat_output_gat:
                    if dim_before_decoder % self.heads != 0:
                        raise ValueError(f"Layer output GAT: dim_before_decoder ({dim_before_decoder}) "
                                         f"harus habis dibagi heads ({self.heads}) jika concat_output_gat=True.")
                    out_ch_layer = dim_before_decoder // self.heads
                else: # Average untuk output
                    out_ch_layer = dim_before_decoder
                concat_layer = self.concat_output_gat
            else: # Untuk layer tersembunyi
                out_ch_layer = self.gat_hidden_channels_per_head
                concat_layer = self.concat_hidden_gat
            
            self.convs.append(
                nng.GATConv(
                    in_channels=current_dim_input,
                    out_channels=out_ch_layer,
                    heads=self.heads,
                    dropout=self.dropout_gat,
                    concat=concat_layer, # Gunakan concat_layer yang sudah ditentukan
                    # add_self_loops=False # Bisa ditambahkan jika edge_index tidak punya self-loops
                )
            )
            
            # Update current_dim_input untuk layer berikutnya
            if concat_layer:
                current_dim_input = out_ch_layer * self.heads
            else:
                current_dim_input = out_ch_layer
        
        self.activation = nn.ELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x)

        for i, conv in enumerate(self.convs):
            is_last_layer = (i == len(self.convs) - 1)
            x = conv(x, edge_index)
            if not is_last_layer: # Aktivasi dan dropout hanya untuk layer tersembunyi
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout_gat, training=self.training)
        
        x = self.decoder(x)
        return x