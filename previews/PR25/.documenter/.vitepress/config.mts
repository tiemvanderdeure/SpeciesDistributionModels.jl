import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import mathjax3 from "markdown-it-mathjax3";
import footnote from "markdown-it-footnote";

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/SpeciesDistributionModels.jl/previews/PR25/',// TODO: replace this in makedocs!
  title: 'SpeciesDistributionModels.jl',
  description: 'Species distirbution modelling in Julia',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }]],
  ignoreDeadLinks: true,

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin),
      md.use(mathjax3),
      md.use(footnote)
    },
    theme: {
      light: "github-light",
      dark: "github-dark"}
  },
  themeConfig: {
    outline: 'deep',
    
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav: [
      { text: 'Getting started', link: '/getting_started' },
      { text: 'Examples', items:
        [
          { text: 'Distribution of Eucalyptus regnans', link: '/eucalyptus_regnans' }, 
        ]
      },
      { text: 'API', link: '/api' },
      { text: 'Ecosystem',
        items: [
          { text: 'Rasters.jl', link: 'https://rafaqz.github.io/Rasters.jl/dev/' },
          { text: 'MLJ.jl', link: 'https://juliaai.github.io/MLJ.jl/dev/' },
          { text: 'GBIF2.jl', link: 'https://rafaqz.github.io/GBIF2.jl/dev/' },
          { text: 'RasterDataSources.jl', link: 'http://docs.ecojulia.org/RasterDataSources.jl/dev/' },
         ]
       },
    ],
    editLink: { pattern: "https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/tiemvanderdeure/SpeciesDistributionModels.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})



